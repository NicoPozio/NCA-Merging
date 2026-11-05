

from dataclasses import dataclass

import torch
import torch.nn.functional as Fn

from .config import (
    W, O, CINDER, CLIPPED, E, F, FLAMMABLE, ICE, H, LAVA, MATERIAL_PRIORITY, N_CHANNELS,
    N_MATERIALS, N_RULES, STONE, R, S, SOIL, STRUCTURES, T, GROUND, C, V, WX, WY, SimParams,
    material_table,
)

#(dy, dx) of the 8 neighbours cells 
OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
#possible wind directions (y is positive going down and, x is positive going right)
WIND_DIRS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
#For every possible wind directions compute the index of the updwind cell
UPWIND_INDEX = tuple(OFFSETS.index((-dy, -dx)) for dy, dx in WIND_DIRS)


def neighbors(x):
    """(B, C, H, W) -> (B, C, H, W, 8), for every cell take the values of neighbours, as they appear
    in OFFSETS

    Padding replicates border, the influence of the replicated cell is not important
    since at every tick the border is rewritten as it was
    Practically for each cell of every channel we have a list of the values of its neighbours
    """
    h, w = x.shape[-2:]
    p = Fn.pad(x, (1, 1, 1, 1), mode="replicate")
    return torch.stack([p[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in OFFSETS], dim=-1)


def border_mask(h, w, device):
    m = torch.zeros(h, w, dtype=torch.bool, device=device)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m

def apply_border(state, t_amb):
    """The border is always ground, it never change"""
    b = border_mask(*state.shape[-2:], state.device)
    fixed = torch.zeros(N_CHANNELS, device=state.device, dtype=state.dtype)
    fixed[GROUND] = 1.0
    fixed[T] = t_amb
    #No in-place writes, so the NCA can backpropagate through this function too
    return torch.cat([torch.where(b, fixed[:WX, None, None], state[:, :WX]), state[:, WX:]], 1)


def inject_events(state, masks, p: SimParams):
    """It applies events in a tick.
    masks: (B, 3, H, W) bool = rain, bolt, ignite"""
    if masks is None:
        return state #No events
    rain, bolt, ignite = masks[:, 0], masks[:, 1], masks[:, 2]
    ch = list(state.unbind(1))
    ch[H] = torch.where(rain, (ch[H] + p.rain_h).clamp(max=1.0), ch[H])
    ch[E] = torch.where(bolt, torch.ones_like(ch[E]), ch[E])
    temp = torch.where(bolt, torch.ones_like(ch[T]), ch[T])
    ch[T] = torch.where(ignite, temp.clamp(min=p.ignite_t), temp)
    ch[F] = torch.where(ignite, torch.ones_like(ch[F]), ch[F])
    return apply_border(torch.stack(ch, 1), p.t_amb)


@dataclass
class RuleOut:
    delta: torch.Tensor                 #(B, 20, H, W) continous increment
    ignite: torch.Tensor = None         #(B, H, W) bool
    extinguish: torch.Tensor = None     #(B, H, W) bool
    mat_target: torch.Tensor = None     #(B, H, W) long, every cell contains the value of the new material
    mat_w: torch.Tensor = None          #(B, H, W) float, new value of W, nan = none
    trigger: torch.Tensor = None        #(B, H, W) bool: cells where a task is triggered in this tick

    def discrete(self):
        out = torch.zeros_like(self.delta[:, 0], dtype=torch.bool)
        if self.ignite is not None:
            out |= self.ignite
        if self.extinguish is not None:
            out |= self.extinguish
        if self.mat_target is not None:
            out |= self.mat_target >= 0
        return out


@dataclass
class StepInfo:
    support: torch.Tensor     #(B, 8, H, W) bool: which tasks have acted on which cell
    conflicts: torch.Tensor   #(B, 2, H, W) bool: mask that tells us in which cell there was a conflict between tasks
                              #2 type of conflict: 2 task want to act on the same cells or sums of effect goes out [0,1]
    triggers: torch.Tensor    #(B, 8) long: in how many cells each rule was triggered in a tick


class Simulator:
    """Simulator G"""

    def __init__(self, params: SimParams = None, device="cpu"):
        self.p = params or SimParams()
        self.device = torch.device(device)
        m = self.p.materials
        self.k = material_table(m.k, self.device)
        #transform inf into a finite value to permit the computation of the product
        self.ign_finite = torch.nan_to_num(material_table(m.ignition, self.device), posinf=2.0)
        self.starting_combustibility = material_table(m.starting_combustibility, self.device)
        self.conductor = material_table(m.conductor, self.device)
        self.absorb = material_table(m.absorbency, self.device)
        self.is_soil = torch.zeros(N_MATERIALS, device=self.device)
        self.is_soil[list(SOIL)] = 1.0
        self.is_flam = torch.zeros(N_MATERIALS, device=self.device)
        self.is_flam[list(FLAMMABLE)] = 1.0
        self.is_struct = torch.zeros(N_MATERIALS, device=self.device)
        self.is_struct[list(STRUCTURES)] = 1.0
        self.rule_fns = (self.t1, self.t2, self.t3, self.t4, self.t5, self.t6, self.t7, self.t8)

    #utility functions

    @staticmethod
    def _prop(mat, table):
        # mat (B, 8, H, W) one-hot -> (B, H, W)
        """Every room is one-hot encoded, where the 8 channels represent the material,
        then we encode a room in a single channel with the value contained in table"""
        return torch.einsum("bmhw,m->bhw", mat, table)

    def _ctx(self, s):
        """Utility functions that compute quantity used by more rules"""
        mat = s[:, :N_MATERIALS] #(B, 8, H, W)
        mat_idx = mat.argmax(1) #(B,H,W) (in every cell we have the index of its material)
        return {
            "mat": mat,
            "idx": mat_idx,
            "soil": self._prop(mat, self.is_soil) > 0.5, #(B, H, W) True where is soil, False where is not
            "flam": self._prop(mat, self.is_flam) > 0.5,
            "struct": self._prop(mat, self.is_struct) > 0.5,
            "n_w": neighbors(s[:, W:W + 1])[:, 0],     # (B, H, W, 8): the value of water in neighbours cell for each cell
            "n_e": neighbors(s[:, E:E + 1])[:, 0],
        }

    @staticmethod
    def _mean_n(x):
        return neighbors(x.unsqueeze(1))[:, 0].mean(-1)

    #Base Model
    def base(self, s, ctx):
        p = self.p #parameters
        d = torch.zeros_like(s)
        temp = s[:, T] #temperature of each cell
        k = self._prop(ctx["mat"], self.k) #how much each cell conducts eat
        nk = neighbors(k.unsqueeze(1))[:, 0]
        nt = neighbors(temp.unsqueeze(1))[:, 0]
        kij = 2 * k.unsqueeze(-1) * nk / (k.unsqueeze(-1) + nk).clamp(min=p.eps) #(B, H, W, 8), how quickly 2 cells tends to go at same T
        d[:, T] = p.heat_rate * (kij * (nt - temp.unsqueeze(-1))).mean(-1) - p.heat_relax * (temp - p.t_amb)

        hum = s[:, H]
        d[:, H] = p.hum_rate * (self._mean_n(hum) - hum) - p.hum_decay * hum

        #Water goes only on soil cell
        w = s[:, W]
        soil = ctx["soil"].float() #tensors of 1.0f and 0.0f
        n_soil = neighbors(soil.unsqueeze(1))[:,0] #(B, H, W, 8)
        cnt = n_soil.sum(-1) #(B,H,W) how many neighbors of a cell are soil
        mean_soil = (n_soil * ctx["n_w"]).sum(-1) / cnt.clamp(min=1) #avg water in neighbours
        spread = torch.where(cnt > 0, p.w_rate * (mean_soil - w), torch.zeros_like(w)) #how much the cell is close to the mean
        d[:, W] = soil * (spread - w.clamp(max=p.w_decay)) 

        vap = s[:, V]
        d[:, V] = p.vap_rate * (self._mean_n(vap) - vap) - p.vap_decay * vap
        smoke = s[:, S]
        d[:, S] = p.smoke_rate * (self._mean_n(smoke) - smoke) - p.smoke_decay * smoke


        #No diffusion, only decay
        d[:, F] = -s[:, F].clamp(max=p.fire_decay)
        d[:, E] = -p.charge_decay * s[:, E]
        d[:, R] = -s[:, R].clamp(max=p.rest_decay)
        return d

    #Eight Tasks

    def t1(self, s, ctx):
        """Combustions"""
        p = self.p
        d = torch.zeros_like(s)
        temp, hum, fuel_c, fire = s[:, T], s[:, H], s[:, C], s[:, F]
        flam = ctx["flam"] #(B, H, W) where is flamable
        thr = self._prop(ctx["mat"], self.ign_finite) #temperature of combusion #(B,H,W)
        has_c = fuel_c > p.eps
        burning = (fire > p.eps) & flam & has_c
        ignite = flam & has_c & (temp >= thr) & (hum <= p.t1_max_h_ignite)
        b = burning.float()
        n_burning = neighbors(b.unsqueeze(1))[:, 0].mean(-1) #share of neighbours on fire
        d[:, C] = -b * fuel_c.clamp(max=p.t1_c_burn)
        d[:, T] = (b * p.t1_heat + p.t1_smoke_heat * s[:, S] * (1 - temp)
                   + p.t1_radiate * n_burning * (1 - temp))
        d[:, S] = b * p.t1_smoke

        burnt = flam & ~has_c
        extinguish = ((fire > p.eps) & (hum >= p.t1_h_extinguish)) | burnt
        target = torch.where(burnt, torch.full_like(ctx["idx"], CINDER), torch.full_like(ctx["idx"], -1))
        return RuleOut(d, ignite=ignite, extinguish=extinguish, mat_target=target,
                       trigger=ignite & (fire <= p.eps))

    def t2(self, s, ctx):
        """Wind moves T,V and S, it goes from the upwind cell to the downind one"""
        p = self.p
        d = torch.zeros_like(s)
        wx, wy = s[:, WX], s[:, WY] #B,H,W
        mag = torch.sqrt(wx ** 2 + wy ** 2).clamp(max=1.0)
        angle = torch.atan2(wy, wx)
        dir_idx = torch.round(angle / (torch.pi / 4)).long() % 8
        up_idx = torch.tensor(UPWIND_INDEX, device=s.device)[dir_idx]           # (B, H, W)
        active = mag > p.eps
        for ch in (T, V, S):
            n = neighbors(s[:, ch:ch + 1])[:, 0]                              # (B, H, W, 8)
            up = n.gather(-1, up_idx.unsqueeze(-1)).squeeze(-1)
            d[:, ch] = torch.where(active, p.t2_rate * mag * (up - s[:, ch]), torch.zeros_like(up))
        return RuleOut(d, trigger=(d[:, (T, V, S)].abs() > p.eps).any(1))

    def t3(self, s, ctx):
        """Water"""
        p = self.p
        d = torch.zeros_like(s)
        w, temp, hum = s[:, W], s[:, T], s[:, H]
        soil = ctx["soil"]
        absorb_k = self._prop(ctx["mat"], self.absorb)
        #a cell's own water wets it fully, water next door only up to t3_h_wet_max
        near = (p.t3_wet * ctx["n_w"].mean(-1)).clamp(max=(p.t3_h_wet_max - hum).clamp(min=0))
        d[:, H] = torch.maximum(p.t3_wet * w, near)
        #water cools only what is hotter than ambient, so a lake does not cool itself to zero and freeze
        d[:, T] = -p.t3_cool * w * (temp - p.t_amb).clamp(min=0)

        puddle = soil & (hum >= p.t3_h_puddle)
        gain = puddle.float() * p.t3_puddle_rate * (1 - absorb_k)
        absorbed = soil.float() * p.t3_absorb_rate * absorb_k * w
        evap = torch.where(temp >= p.t3_t_evap, (w - absorbed).clamp(max=p.t3_evap), torch.zeros_like(w))
        remaining = w - absorbed - evap
        thin = soil & (w > 0) & (w < p.t3_thin) & ~puddle
        vanish = thin.float() * remaining

        d[:, W] = gain - absorbed - evap - vanish
        d[:, H] = d[:, H] + absorbed
        d[:, V] = evap
        return RuleOut(d, trigger=gain > p.eps)

    def t4(self, s, ctx):
        """Electricity"""
        p = self.p
        d = torch.zeros_like(s)
        e, rest = s[:, E], s[:, R]
        conductor = ((self._prop(ctx["mat"], self.conductor) > 0.5) | (s[:, W] >= p.t4_w_conduct))
        conductor &= ctx["idx"] != ICE
        discharge = conductor & (rest <= p.eps) & (e < p.t4_e_on) & (ctx["n_e"].max(-1).values >= p.t4_e_on)
        charged = e >= p.t4_e_on
        n_charged = neighbors(charged.float().unsqueeze(1))[:, 0].mean(-1)
        d[:, E] = discharge.float() * (1 - e)
        d[:, R] = charged.float() * (1 - rest)
        d[:, T] = p.t4_heat * e + p.t4_radiate * n_charged * (1 - s[:, T])
        return RuleOut(d, trigger=discharge)

    def t5(self, s, ctx):
        """Ice"""
        p = self.p
        d = torch.zeros_like(s)
        freeze = ctx["soil"] & (s[:, W] >= p.t5_w_freeze) & (s[:, T] <= p.t5_t_freeze)
        melt = (ctx["idx"] == ICE) & (s[:, T] >= p.t5_t_melt)
        target = torch.full_like(ctx["idx"], -1)
        target = torch.where(freeze, torch.full_like(target, ICE), target)
        target = torch.where(melt, torch.full_like(target, GROUND), target)
        mat_w = torch.full_like(s[:, W], float("nan"))
        mat_w = torch.where(melt, torch.ones_like(mat_w), mat_w)
        return RuleOut(d, mat_target=target, mat_w=mat_w, trigger=freeze | melt)

    def t6(self, s, ctx):
        """Oil"""
        p = self.p
        d = torch.zeros_like(s)
        fuel, fire = s[:, O], s[:, F]
        temp = s[:, T]
        burning = (fire > p.eps) & (fuel > p.eps)
        ignite = (fuel >= p.t6_o_ignite) & (temp >= p.t6_t_ignite) & (s[:, H] <= p.t6_max_h_ignite)
        b = burning.float()
        n_burning = neighbors(b.unsqueeze(1))[:, 0].mean(-1)
        d[:, O] = -b * fuel.clamp(max=p.t6_o_burn)
        d[:, T] = b * p.t6_heat + p.t6_radiate * n_burning * (1 - s[:, T])
        d[:, S] = b * p.t6_smoke
        return RuleOut(d, ignite=ignite, trigger=ignite & (fire <= p.eps))

    def t7(self, s, ctx):
        """Lava"""
        p = self.p
        d = torch.zeros_like(s)
        lava = ctx["idx"] == LAVA
        temp = s[:, T]
        n_lava = neighbors(lava.float().unsqueeze(1))[:, 0].mean(-1)
        d[:, T] = -lava.float() * temp.clamp(max=p.t7_cool) + p.t7_radiate * n_lava * (1 - temp) * (~lava).float()
        contact = lava & (ctx["n_w"].max(-1).values >= p.t7_w_contact)
        solid = lava & ((temp < p.t7_t_solid) | contact)
        #only lava melts stone: fire, fuel and lightning also bring T near 1, but must not make lava
        melt = (ctx["idx"] == STONE) & (temp >= p.t7_t_melt) & (n_lava > 0)
        d[:, V] = contact.float() * p.t7_vapor
        target = torch.full_like(ctx["idx"], -1)
        target = torch.where(solid, torch.full_like(target, STONE), target)
        target = torch.where(melt, torch.full_like(target, LAVA), target)
        return RuleOut(d, mat_target=target, trigger=solid | melt)

    def t8(self, s, ctx):
        """Vapor"""
        p = self.p
        d = torch.zeros_like(s)
        cond = (s[:, V] >= p.t8_v_cond) & (s[:, T] <= p.t8_t_cond)
        #gradual condensation: every tick a fixed share of the vapor turns into humidity
        condensed = cond.float() * p.t8_rate * s[:, V]
        d[:, V] = -condensed
        d[:, H] = p.t8_h_gain * condensed
        return RuleOut(d, trigger=cond)



    def rule_outputs(self, s, active):
        """Increase of state, coming from base model and various tasks"""
        ctx = self._ctx(s)
        d_base = self.base(s, ctx)
        outs = []
        for i, fn in enumerate(self.rule_fns):
            out = fn(s, ctx)
            a = active[:, i].view(-1, 1, 1)
            out.delta = out.delta * a.unsqueeze(1)
            for name in ("ignite", "extinguish", "trigger"):
                val = getattr(out, name)
                if val is not None:
                    setattr(out, name, val & a)
            if out.mat_target is not None:
                out.mat_target = torch.where(a, out.mat_target, torch.full_like(out.mat_target, -1))
            outs.append(out)
        return d_base, outs

    def step(self, state, active, event_masks=None, diagnostics=True):
        """state (B, 20, H, W), active (B, 8) bool, event_masks (B, 3, H, W) o None.

        Restituisce lo stato al tick successivo e le informazioni per la diagnosi
        (con diagnostics=False solo i trigger, che costano poco).
        """
        p = self.p
        s = inject_events(state, event_masks, p)
        active = active.to(torch.bool)
        d_base, outs = self.rule_outputs(s, active)
        deltas = torch.stack([o.delta for o in outs], 1)                     # (B, 8, 20, H, W)
        cont = s + d_base + deltas.sum(1)

        # richieste discrete
        ign = torch.zeros_like(s[:, 0], dtype=torch.bool)
        ext = torch.zeros_like(ign)
        for o in outs:
            if o.ignite is not None:
                ign |= o.ignite
            if o.extinguish is not None:
                ext |= o.extinguish

        new_idx = torch.full_like(ign, -1, dtype=torch.long)
        new_aq = torch.full_like(s[:, W], float("nan"))
        for i in sorted(MATERIAL_PRIORITY, key=MATERIAL_PRIORITY.get, reverse=True):
            o = outs[i]
            req = o.mat_target >= 0
            take = req & (new_idx < 0)
            new_idx = torch.where(take, o.mat_target, new_idx)
            if o.mat_w is not None:
                new_aq = torch.where(take, o.mat_w, new_aq)

        # 5. taglio: lo applichiamo prima dei cambi discreti, che scrivono comunque valori in [0, 1]
        nxt = cont.clone()
        nxt[:, CLIPPED] = cont[:, CLIPPED].clamp(0.0, 1.0)

        # 4. fuoco: lo spegnimento vince sull'accensione
        nxt[:, F] = torch.where(ign, torch.ones_like(nxt[:, F]), nxt[:, F])
        nxt[:, F] = torch.where(ext, torch.zeros_like(nxt[:, F]), nxt[:, F])

        # 4. materiale: one-hot nuovo, U del materiale nuovo, strati azzerati sulle strutture
        changed = new_idx >= 0
        if changed.any():
            tgt = new_idx.clamp(min=0)
            onehot = Fn.one_hot(tgt, N_MATERIALS).permute(0, 3, 1, 2).to(s.dtype)
            ch = changed.unsqueeze(1)
            nxt[:, :N_MATERIALS] = torch.where(ch, onehot, nxt[:, :N_MATERIALS])
            nxt[:, C] = torch.where(changed, self.starting_combustibility[tgt], nxt[:, C])
            to_struct = changed & (self.is_struct[tgt] > 0.5)
            nxt[:, W] = torch.where(to_struct, torch.zeros_like(nxt[:, W]), nxt[:, W])
            nxt[:, O] = torch.where(to_struct, torch.zeros_like(nxt[:, O]), nxt[:, O])
            set_aq = changed & ~torch.isnan(new_aq)
            nxt[:, W] = torch.where(set_aq, torch.nan_to_num(new_aq), nxt[:, W])

        nxt[:, WX:] = s[:, WX:]
        nxt = apply_border(nxt, p.t_amb)

        if diagnostics:
            info = self._diagnostics(s, d_base, outs, deltas, cont)
        else:
            b = border_mask(*s.shape[-2:], s.device)
            info = StepInfo(None, None, torch.stack([(o.trigger & ~b).flatten(1).sum(1) for o in outs], 1))
        return nxt, info

    def _diagnostics(self, s, d_base, outs, deltas, cont):
        """Maschera dei supporti e registro dei conflitti.

        Il registro marca esattamente le celle in cui G può differire dalla somma
        O = y0 + sum_i (y_i - y0), con y_i = un tick del solo base + regola i:
        - canale 0: due regole con richieste discrete sulla stessa cella (priorità o doppia richiesta),
          oppure un cambio discreto che sovrascrive un canale incrementato da un'altra regola;
        - canale 1: un canale incrementato da almeno due regole, dove il taglio a [0, 1] è scattato
          nella somma o in uno dei singoli.
        """
        p = self.p
        eps = p.eps
        touched = deltas.abs() > eps                                             # (B, 8, 20, H, W)
        discrete = torch.stack([o.discrete() for o in outs], 1)                  # (B, 8, H, W)
        support = touched.any(2) | discrete

        # canale 0
        n_discrete = discrete.float().sum(1)
        overwritten = torch.zeros_like(touched[:, :, 0])                         # (B, 8, H, W)
        for i, o in enumerate(outs):
            chans = []
            if o.ignite is not None or o.extinguish is not None:
                chans.append(F)
            if o.mat_target is not None:
                chans += [C, W, O]
            if not chans:
                continue
            others = torch.cat([touched[:, :i], touched[:, i + 1:]], 1)[:, :, chans].any(1).any(1)
            overwritten[:, i] = discrete[:, i] & others
        conflict_discrete = (n_discrete >= 2) | overwritten.any(1)

        # canale 1
        base_next = s + d_base
        out_of_range = lambda x: (x < -eps) | (x > 1 + eps)
        clip_any = out_of_range(cont[:, CLIPPED]) | out_of_range(base_next[:, CLIPPED])
        single = base_next.unsqueeze(1) + deltas                                 # (B, 8, 20, H, W)
        clip_any = clip_any | out_of_range(single[:, :, CLIPPED]).any(1)
        multi = touched[:, :, CLIPPED].sum(1) >= 2
        conflict_clip = (clip_any & multi).any(1)

        b = border_mask(*s.shape[-2:], s.device)
        support = support & ~b
        conflicts = torch.stack([conflict_discrete, conflict_clip], 1) & ~b
        triggers = torch.stack([(o.trigger & ~b).flatten(1).sum(1) for o in outs], 1)
        return StepInfo(support=support, conflicts=conflicts, triggers=triggers)

    @torch.no_grad()
    def rollout(self, state, active, steps, event_masks=None, keep_info=True):
        """Fa girare il simulatore per steps tick.

        event_masks: (B, steps, 3, H, W) o None. Restituisce la traiettoria (B, steps + 1, 20, H, W)
        e, se keep_info, supporti (B, steps, 8, H, W), conflitti (B, steps, 2, H, W) e trigger (B, 8)
        accumulati su tutta la finestra.
        """
        traj = [state]
        sup, conf = [], []
        trig = torch.zeros(state.shape[0], N_RULES, dtype=torch.long, device=state.device)
        x = state
        for t in range(steps):
            m = None if event_masks is None else event_masks[:, t]
            x, info = self.step(x, active, m, diagnostics=keep_info)
            traj.append(x)
            trig += info.triggers
            if keep_info:
                sup.append(info.support)
                conf.append(info.conflicts)
        traj = torch.stack(traj, 1)
        if not keep_info:
            return traj, None, None, trig
        return traj, torch.stack(sup, 1), torch.stack(conf, 1), trig