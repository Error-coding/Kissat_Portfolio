# Credit for this wrapper of the configuration space of Kissat 2024 goes to Dr. Markus Iser

from ConfigSpace import ConfigurationSpace, Integer, InCondition


def gate_options():
    # extract gates in variable elimination
    extract = Integer("extract", (0, 1), default=1)
    extract_and = Integer("ands", (0, 1), default=1)
    extract_equiv = Integer("equivalences", (0, 1), default=1)
    extract_ite = Integer("ifthenelse", (0, 1), default=1)
    extract_def = Integer("definitions", (0, 1), default=1)
    def_ticks = Integer("definitionticks", (0, 2147483647), default=1000000)
    def_cores = Integer("definitioncores", (1, 100), default=2)
    options = [ extract, extract_and, extract_equiv, extract_ite, extract_def, def_ticks, def_cores ]
    
    conditions = []
    for child in [extract_and, extract_equiv, extract_ite, extract_def]:
        conditions.append(InCondition(child, extract, [1])) # child implies parent
        # todo: parent implies at-least-one(child)
    conditions.append(InCondition(def_ticks, extract_def, [1]))
    conditions.append(InCondition(def_cores, extract_def, [1]))
    
    # congruence closure on extracted gates
    congruence = Integer("congruence", (0, 1), default=1)
    congruence_ands = Integer("congruenceands", (0, 1), default=1)
    congruence_ites = Integer("congruenceites", (0, 1), default=1)
    congruence_xors = Integer("congruencexors", (0, 1), default=1)
    congruence_binaries = Integer("congruencebinaries", (0, 1), default=1)
    congruence_once = Integer("congruenceonce", (0, 1), default=0)
    congruence_andarity = Integer("congruenceandarity", (2, 50000000), default=1000000)
    congruence_xorarity = Integer("congruencexorarity", (2, 20), default=4)
    congruence_xorcounts = Integer("congruencexorcounts", (1, 2147483647), default=2)
    options += [ congruence, congruence_ands, congruence_ites, congruence_xors, congruence_binaries, congruence_once, congruence_andarity, congruence_xorarity, congruence_xorcounts ]
    
    for child in [congruence_ands, congruence_ites, congruence_xors, congruence_binaries, congruence_once]:
        conditions.append(InCondition(child, congruence, [1]))
    # todo: congurence implies at-least-one-of(congruence_ands, congruence_ites, congruence_xors))
    conditions.append(InCondition(congruence_andarity, congruence_ands, [1]))
    conditions.append(InCondition(congruence_xorarity, congruence_xors, [1]))
    conditions.append(InCondition(congruence_xorcounts, congruence_xors, [1]))
    
    return options + conditions


def phasing_options():
    # enable stable search mode 0=focused,1=switch,2=stable
    stable = Integer("stable", (0, 2), default=1)
    # mode switch limit and interval
    modeinit = Integer("modeinit", (10, 100000000), default=1000)
    modeint = Integer("modeint", (10, 100000000), default=1000)
    
    conditions = []
    conditions.append(InCondition(modeinit, stable, [1]))
    conditions.append(InCondition(modeint, stable, [1]))
    
    phase = Integer("phase", (0, 1), default=1) # initial phase
    target = Integer("target", (0, 2), default=1) # (1=stable mode only, 2=also in focused mode)
    phasesaving = Integer("phasesaving", (0, 1), default=1)
    # todo: condition target=2 disables phasesaving
    
    options = [stable, modeinit, modeint, phase, target, phasesaving]
    
    # initialize phases by unit propagation
    warmup = Integer("warmup", (0, 1), default=1)
    options.append(warmup)
        
    # try some lucky assignments
    lucky = Integer("lucky", (0, 1), default=1)
    luckyearly = Integer("luckyearly", (0, 1), default=1)
    luckylate = Integer("luckylate", (0, 1), default=1)
    options += [lucky, luckyearly, luckylate]
    conditions.append(InCondition(luckyearly, lucky, [1]))
    conditions.append(InCondition(luckylate, lucky, [1]))
    
    # reinitialization of decision phases
    rephase = Integer("rephase", (0, 1), default=1)
    rephaseinit = Integer("rephaseinit", (10, 100000), default=1000)
    rephaseint = Integer("rephaseint", (10, 100000), default=1000)
    options += [rephase, rephaseinit, rephaseint]
    conditions.append(InCondition(rephaseinit, rephase, [1]))
    conditions.append(InCondition(rephaseint, rephase, [1]))
    
    return options + conditions


def decision_options():
    # enable variable bumping
    bump = Integer("bump", (0, 1), default=1)
    bumpreasons = Integer("bumpreasons", (0, 1), default=1)
    bumpreasonslimit = Integer("bumpreasonslimit", (1, 2147483647), default=10)
    bumpreasonsrate = Integer("bumpreasonsrate", (1, 2147483647), default=10)
    decay = Integer("decay", (1, 200), default=50)
    
    options = [bump, bumpreasons, bumpreasonslimit, bumpreasonsrate, decay]
    conditions = []
    conditions.append(InCondition(bumpreasons, bump, [1]))
    conditions.append(InCondition(bumpreasonslimit, bumpreasons, [1]))
    conditions.append(InCondition(bumpreasonsrate, bumpreasons, [1]))
    conditions.append(InCondition(decay, bump, [1]))
    
    # random decisions
    randec = Integer("randec", (0, 1), default=1)
    randecfocused = Integer("randecfocused", (0, 1), default=1)
    randecstable = Integer("randecstable", (0, 1), default=0)
    randeclength = Integer("randeclength", (1, 2147483647), default=10)
    options += [randec, randecfocused, randecstable, randeclength]
    conditions.append(InCondition(randecfocused, randec, [1]))
    conditions.append(InCondition(randecstable, randec, [1]))
    conditions.append(InCondition(randeclength, randec, [1]))
    
    # reorder decisions (1=stable-mode-only)
    reorder = Integer("reorder", (0, 2), default=2)
    reordermaxsize = Integer("reordermaxsize", (2, 256), default=100)
    options += [reorder, reordermaxsize]
    conditions.append(InCondition(reordermaxsize, reorder, [1]))
    
    # allow chronological backtracking by setting a backjump limit
    chrono = Integer("chrono", (0, 1), default=1)
    chronolevels = Integer("chronolevels", (0, 2147483647), default=100)
    options += [chrono, chronolevels]
    conditions.append(InCondition(chronolevels, chrono, [1]))
    
    return options + conditions

    
def learnts_options():
    # eagerly subsume previous learned clauses
    eagersubsume = Integer("eagersubsume", (0, 4), default=4)
    # jump binary reasons
    jumpreasons = Integer("jumpreasons", (0, 1), default=1)
    # learned clause minimization
    minimize = Integer("minimize", (0, 1), default=1)
    minimizedepth = Integer("minimizedepth", (1, 1000000), default=1000)
    minimizeticks = Integer("minimizeticks", (0, 1), default=1)
    # on-the-fly strengthening
    otfs = Integer("otfs", (0, 1), default=1)
    # promote clauses
    promote = Integer("promote", (0, 1), default=1)
    # learned clauses (1=bin,2=lrg,3=rec)
    shrink = Integer("shrink", (0, 3), default=3)
    
    options = [eagersubsume, jumpreasons, minimize, minimizedepth, minimizeticks, otfs, promote, shrink]
    conditions = []
    conditions.append(InCondition(minimizedepth, minimize, [1]))
    conditions.append(InCondition(minimizeticks, minimize, [1]))
    
    # learned clause tier glue limits
    tier1 = Integer("tier1", (1, 100), default=2)
    tier1relative = Integer("tier1relative", (0, 1000), default=500)
    tier2 = Integer("tier2", (1, 1000), default=6)
    tier2relative = Integer("tier2relative", (0, 1000), default=900)
    options += [tier1, tier1relative, tier2, tier2relative]
    
    # learned clause reduction
    reducehigh = Integer("reducehigh", (0, 1000), default=900)
    reducelow = Integer("reducelow", (0, 1000), default=500)
    options += [reducehigh, reducelow]
    
    return options + conditions

    
def restart_options():
    # enable restarts
    restart = Integer("restart", (0, 1), default=1)
    restartint = Integer("restartint", (1, 10000), default=1)
    restartmargin = Integer("restartmargin", (0, 25), default=10)
    restartreusetrail = Integer("restartreusetrail", (0, 1), default=1)
    
    options = [restart, restartint, restartmargin, restartreusetrail]
    conditions = []
    conditions.append(InCondition(restartint, restart, [1]))
    conditions.append(InCondition(restartmargin, restart, [1]))
    conditions.append(InCondition(restartreusetrail, restart, [1]))
    
    # stable reluctant doubling restarting
    reluctant = Integer("reluctant", (0, 1), default=1)
    reluctantint = Integer("reluctantint", (2, 32768), default=1024)
    reluctantlim = Integer("reluctantlim", (0, 1073741824), default=1048576)
    
    options += [reluctant, reluctantint, reluctantlim]
    conditions.append(InCondition(reluctantint, reluctant, [1]))
    conditions.append(InCondition(reluctantlim, reluctant, [1]))
    
    # fast and slow exponential moving average window
    emafast = Integer("emafast", (10, 1000000), default=33)
    # emaslow = Integer("emaslow", (100, 1000000), default=100000) # used in more than one place (not just restarts), condition: slow > fast
    options.append(emafast)
    
    return options + conditions
    
    
def pre_and_inprocessing_options():
    # enable probing
    probe = Integer("probe", (0, 1), default=1)
    proberounds = Integer("proberounds", (1, 2147483647), default=2)
    options = [probe, proberounds]
    conditions = []
    conditions.append(InCondition(proberounds, probe, [1]))
    
    # binary clause backbone (2=eager)
    backbone = Integer("backbone", (0, 2), default=1)
    backbonemaxrounds = Integer("backbonemaxrounds", (1, 2147483647), default=1000)
    backbonerounds = Integer("backbonerounds", (1, 2147483647), default=100)
    options += [backbone, backbonemaxrounds, backbonerounds]
    conditions.append(InCondition(backbonemaxrounds, backbone, [1]))
    conditions.append(InCondition(backbonerounds, backbone, [1]))
    
    # transitive reduction of binary clauses
    transitive = Integer("transitive", (0, 1), default=1)
    transitivekeep = Integer("transitivekeep", (0, 1), default=1)
    options += [transitive, transitivekeep]
    conditions.append(InCondition(transitivekeep, transitive, [1]))
    
    # equivalent literal substitution
    substitute = Integer("substitute", (0, 1), default=1)
    substituteeffort = Integer("substituteeffort", (1, 1000), default=10)
    substituterounds = Integer("substituterounds", (1, 100), default=2)
    options += [substitute, substituteeffort, substituterounds]
    conditions.append(InCondition(substituteeffort, substitute, [1]))
    conditions.append(InCondition(substituterounds, substitute, [1]))
    
    # vivification
    vivify = Integer("vivify", (0, 1), default=1)
    vivifyfocusedtiers = Integer("vivifyfocusedtiers", (0, 1), default=1)
    vivifyirr = Integer("vivifyirr", (0, 100), default=3)
    vivifysort = Integer("vivifysort", (0, 1), default=1)
    vivifytier1 = Integer("vivifytier1", (0, 100), default=3)
    vivifytier2 = Integer("vivifytier2", (0, 100), default=3)
    vivifytier3 = Integer("vivifytier3", (0, 100), default=1)
    options += [vivify, vivifyfocusedtiers, vivifyirr, vivifysort, vivifytier1, vivifytier2, vivifytier3]
    conditions.append(InCondition(vivifyfocusedtiers, vivify, [1]))
    conditions.append(InCondition(vivifyirr, vivify, [1]))
    conditions.append(InCondition(vivifysort, vivify, [1]))
    conditions.append(InCondition(vivifytier1, vivify, [1]))
    conditions.append(InCondition(vivifytier2, vivify, [1]))
    conditions.append(InCondition(vivifytier3, vivify, [1]))
    
    # enable SAT sweeping
    sweep = Integer("sweep", (0, 1), default=1)
    sweepclauses = Integer("sweepclauses", (0, 2147483647), default=1024)
    sweepcomplete = Integer("sweepcomplete", (0, 1), default=0)
    sweepdepth = Integer("sweepdepth", (0, 2147483647), default=2)
    sweepfliprounds = Integer("sweepfliprounds", (0, 2147483647), default=1)
    sweepmaxclauses = Integer("sweepmaxclauses", (2, 2147483647), default=32768)
    sweepmaxdepth = Integer("sweepmaxdepth", (1, 2147483647), default=3)
    sweepmaxvars = Integer("sweepmaxvars", (2, 2147483647), default=8192)
    sweeprand = Integer("sweeprand", (0, 1), default=0)
    sweepvars = Integer("sweepvars", (0, 2147483647), default=256)
    options += [sweep, sweepclauses, sweepcomplete, sweepdepth, sweepfliprounds, sweepmaxclauses, sweepmaxdepth, sweepmaxvars, sweeprand, sweepvars]
    conditions.append(InCondition(sweepclauses, sweep, [1]))
    conditions.append(InCondition(sweepcomplete, sweep, [1]))
    conditions.append(InCondition(sweepdepth, sweep, [1]))
    conditions.append(InCondition(sweepfliprounds, sweep, [1]))
    conditions.append(InCondition(sweepmaxclauses, sweep, [1]))
    conditions.append(InCondition(sweepmaxdepth, sweep, [1]))
    conditions.append(InCondition(sweepmaxvars, sweep, [1]))
    conditions.append(InCondition(sweeprand, sweep, [1]))
    conditions.append(InCondition(sweepvars, sweep, [1]))    
    
    # initial preprocessing
    preprocess = Integer("preprocess", (0, 1), default=1)
    preprocessbackbone = Integer("preprocessbackbone", (0, 1), default=1)
    preprocesscongruence = Integer("preprocesscongruence", (0, 1), default=1)
    preprocessfactor = Integer("preprocessfactor", (0, 1), default=1)
    preprocessprobe = Integer("preprocessprobe", (0, 1), default=1)
    preprocessrounds = Integer("preprocessrounds", (1, 2147483647), default=1)
    preprocessweep = Integer("preprocessweep", (0, 1), default=1)
    options += [preprocess, preprocessbackbone, preprocesscongruence, preprocessfactor, preprocessprobe, preprocessrounds, preprocessweep]
    conditions.append(InCondition(preprocessbackbone, preprocess, [1]))
    conditions.append(InCondition(preprocesscongruence, preprocess, [1]))
    conditions.append(InCondition(preprocessfactor, preprocess, [1]))
    conditions.append(InCondition(preprocessprobe, preprocess, [1]))
    conditions.append(InCondition(preprocessrounds, preprocess, [1]))
    conditions.append(InCondition(preprocessweep, preprocess, [1]))    
    
    # initial fast variable elimination
    fastel = Integer("fastel", (0, 1), default=1)
    fastelclslim = Integer("fastelclslim", (1, 2147483647), default=100)
    fastelim = Integer("fastelim", (1, 1000), default=8)
    fasteloccs = Integer("fasteloccs", (1, 1000), default=100)
    fastelrounds = Integer("fastelrounds", (1, 1000), default=4)
    fastelsub = Integer("fastelsub", (0, 1), default=1)
    options += [fastel, fastelclslim, fastelim, fasteloccs, fastelrounds, fastelsub]
    conditions.append(InCondition(fastelclslim, fastel, [1]))
    conditions.append(InCondition(fastelim, fastel, [1]))
    conditions.append(InCondition(fasteloccs, fastel, [1]))
    conditions.append(InCondition(fastelrounds, fastel, [1]))
    conditions.append(InCondition(fastelsub, fastel, [1]))
    
    # bounded variable addition
    factor = Integer("factor", (0, 1), default=1)
    factorcandrounds = Integer("factorcandrounds", (0, 2147483647), default=2)
    factorhops = Integer("factorhops", (1, 10), default=3)
    factoriniticks = Integer("factoriniticks", (1, 1000000), default=700)
    factorsize = Integer("factorsize", (2, 2147483647), default=5)
    factorstructural = Integer("factorstructural", (0, 1), default=0)
    options += [factor, factorcandrounds, factorhops, factoriniticks, factorsize, factorstructural]
    conditions.append(InCondition(factorcandrounds, factor, [1]))
    conditions.append(InCondition(factorhops, factor, [1]))
    conditions.append(InCondition(factoriniticks, factor, [1]))
    conditions.append(InCondition(factorsize, factor, [1]))
    conditions.append(InCondition(factorstructural, factor, [1]))    
    
    # bounded variable elimination BVE
    eliminate = Integer("eliminate", (0, 1), default=1)
    eliminatebound = Integer("eliminatebound", (0, 8192), default=16)
    eliminateclslim = Integer("eliminateclslim", (1, 2147483647), default=100)
    eliminateinit = Integer("eliminateinit", (0, 2147483647), default=500)
    eliminateocclim = Integer("eliminateocclim", (0, 2147483647), default=2000)
    eliminaterounds = Integer("eliminaterounds", (1, 10000), default=2)
    options += [eliminate, eliminatebound, eliminateclslim, eliminateinit, eliminateocclim, eliminaterounds]
    conditions.append(InCondition(eliminatebound, eliminate, [1]))
    conditions.append(InCondition(eliminateclslim, eliminate, [1]))
    conditions.append(InCondition(eliminateinit, eliminate, [1]))
    conditions.append(InCondition(eliminateocclim, eliminate, [1]))
    conditions.append(InCondition(eliminaterounds, eliminate, [1]))
    
    # forward subsumption in BVE
    forward = Integer("forward", (0, 1), default=1)
    subsumeclslim = Integer("subsumeclslim", (1, 2147483647), default=1000)
    subsumeocclim = Integer("subsumeocclim", (0, 2147483647), default=1000)
    options += [forward, subsumeclslim, subsumeocclim]
    conditions.append(InCondition(subsumeclslim, forward, [1]))
    conditions.append(InCondition(subsumeocclim, forward, [1]))
    
    return options + conditions

    
def get_kissat2024_confspace() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)
    cs.add(gate_options())
    cs.add(phasing_options())
    cs.add(decision_options())
    cs.add(learnts_options())
    cs.add(restart_options())
    cs.add(pre_and_inprocessing_options())
    return cs