from ConfigSpace import ConfigurationSpace, Integer, InCondition



def getBackboneOpt():
    backbonemaxrounds = Integer("backbonemaxrounds", (1, 100000), default=1000, log=True)
    backbonerounds = Integer("backbonerounds", (1, 10000), default=100, log=True)
    return [backbonemaxrounds, backbonerounds]

def getBumpOpt():
    bumpreasons = Integer("bumpreasons", (0, 1), default=1)
    decay = Integer("decay", (1, 200), default=50)

    return [bumpreasons, decay]

def getChronoOpt():
    return [Integer("chronolevels", (1, 100000), default=100, log=True)]

def getCongruenceOpt():
    congruence_ands = Integer("congruenceands", (0, 1), default=1)
    congruence_ites = Integer("congruenceites", (0, 1), default=1)
    congruence_xors = Integer("congruencexors", (0, 1), default=1)
    congruence_binaries = Integer("congruencebinaries", (0, 1), default=1)
    congruence_once = Integer("congruenceonce", (0, 1), default=0)
    return [congruence_ands, congruence_ites, congruence_xors, congruence_binaries, congruence_once]

def getEliminateOpt():
    eliminatebound = Integer("eliminatebound", (0, 8192), default=16)
    eliminateclslim = Integer("eliminateclslim", (1, 100000), default=100, log=True)
    eliminateinit = Integer("eliminateinit", (1, 500000), default=500, log=True)
    eliminateocclim = Integer("eliminateocclim", (1, 2000000), default=2000, log=True)
    eliminaterounds = Integer("eliminaterounds", (1, 10000), default=2, log=True)
    return [eliminatebound, eliminateclslim, eliminateinit, eliminateocclim, eliminaterounds]

def getExtractOpt():
    extract_and = Integer("ands", (0, 1), default=1)
    extract_equiv = Integer("equivalences", (0, 1), default=1)
    extract_ite = Integer("ifthenelse", (0, 1), default=1)
    extract_def = Integer("definitions", (0, 1), default=1)
    return [extract_and, extract_equiv, extract_ite, extract_def]

def getFactorOpt():
    factorcandrounds = Integer("factorcandrounds", (1, 1000), default=2, log=True)
    factorhops = Integer("factorhops", (1, 10), default=3)
    factoriniticks = Integer("factoriniticks", (1, 70000), default=700, log=True)
    factorsize = Integer("factorsize", (2, 5000), default=5)
    factorstructural = Integer("factorstructural", (0, 1), default=0)
    return [factorcandrounds, factorhops, factoriniticks, factorsize, factorstructural]

def getFastelOpt():
    fastelclslim = Integer("fastelclslim", (1, 10000), default=100, log=True)
    fastelim = Integer("fastelim", (1, 1000), default=8)
    fasteloccs = Integer("fasteloccs", (1, 1000), default=100)
    fastelrounds = Integer("fastelrounds", (1, 1000), default=4)
    fastelsub = Integer("fastelsub", (0, 1), default=1)
    return [fastelclslim, fastelim, fasteloccs, fastelrounds, fastelsub]

def getForwardOpt():
    subsumeclslim = Integer("subsumeclslim", (1, 1000000), default=1000, log=True)
    subsumeocclim = Integer("subsumeocclim", (1, 1000000), default=1000, log=True)
    return [subsumeclslim, subsumeocclim]

# no second level parameters for phase/phasesaving

def getLuckyOpt():
    luckyearly = Integer("luckyearly", (0, 1), default=1)
    luckylate = Integer("luckylate", (0, 1), default=1)
    return [luckyearly, luckylate]

def getPreprocessOpt():
    preprocessbackbone = Integer("preprocessbackbone", (0, 1), default=1)
    preprocesscongruence = Integer("preprocesscongruence", (0, 1), default=1)
    preprocessfactor = Integer("preprocessfactor", (0, 1), default=1)
    preprocessprobe = Integer("preprocessprobe", (0, 1), default=1)
    preprocessrounds = Integer("preprocessrounds", (1, 1000), default=1)
    preprocessweep = Integer("preprocessweep", (0, 1), default=1)
    return [preprocessbackbone, preprocesscongruence, preprocessfactor, preprocessprobe, preprocessrounds, preprocessweep]

def getProbeOpt():
    proberounds = Integer("proberounds", (1, 1000), default=2)
    return [proberounds]

def getRandecOpt():
    randecfocused = Integer("randecfocused", (0, 1), default=1)
    randecstable = Integer("randecstable", (0, 1), default=0)
    randeclength = Integer("randeclength", (1, 10000), default=10, log=True)
    return [randecfocused, randecstable, randeclength]

def getReluctantOpt():
    reluctantint = Integer("reluctantint", (2, 32768), default=1024)
    reluctantlim = Integer("reluctantlim", (1, 1073741824), default=1048576, log=True)
    return [reluctantint, reluctantlim]

def getReorderOpt():
    reordermaxsize = Integer("reordermaxsize", (2, 256), default=100)
    return [reordermaxsize]

def getRephaseOpt():
    rephaseinit = Integer("rephaseinit", (10, 100000), default=1000, log=True)
    rephaseint = Integer("rephaseint", (10, 100000), default=1000, log=True)
    return [rephaseinit, rephaseint]

def getRestartOpt():
    restartint = Integer("restartint", (1, 10000), default=1, log=True)
    restartmargin = Integer("restartmargin", (0, 25), default=10)
    restartreusetrail = Integer("restartreusetrail", (0, 1), default=1)
    return [restartint, restartmargin, restartreusetrail]

def getStableOpt():
    modeinit = Integer("modeinit", (10, 100000000), default=1000, log=True)
    modeint = Integer("modeint", (10, 100000000), default=1000, log=True)
    return [modeinit, modeint]

def getSubstituteOpt():
    substituteeffort = Integer("substituteeffort", (1, 1000), default=10, log=True)
    substituterounds = Integer("substituterounds", (1, 100), default=2)
    return [substituteeffort, substituterounds]

def getSweepOpt():
    sweepclauses = Integer("sweepclauses", (1, 1024 *1024), default=1024, log=True)
    sweepcomplete = Integer("sweepcomplete", (0, 1), default=0)
    sweepdepth = Integer("sweepdepth", (1, 2000), default=2, log=True)
    sweepfliprounds = Integer("sweepfliprounds", (1, 2000), default=1, log=True)
    sweepmaxclauses = Integer("sweepmaxclauses", (2, 32768 * 32768), default=32768, log=True)
    sweepmaxdepth = Integer("sweepmaxdepth", (1, 3000), default=3)
    sweepmaxvars = Integer("sweepmaxvars", (2, 8192 * 8192), default=8192, log=True)
    sweeprand = Integer("sweeprand", (0, 1), default=0)
    sweepvars = Integer("sweepvars", (1, 256 * 1024), default=256, log=True)
    return [sweepclauses, sweepcomplete, sweepdepth, sweepfliprounds, sweepmaxclauses, sweepmaxdepth, sweepmaxvars, sweeprand, sweepvars]

# no secondary options for target

def getTransitiveOpt():
    transitivekeep = Integer("transitivekeep", (0, 1), default=1)
    return [transitivekeep]

def getVivifyOpt():
    vivifyfocusedtiers = Integer("vivifyfocusedtiers", (0, 1), default=1)
    vivifyirr = Integer("vivifyirr", (0, 100), default=3)
    vivifysort = Integer("vivifysort", (0, 1), default=1)
    vivifytier1 = Integer("vivifytier1", (0, 100), default=3)
    vivifytier2 = Integer("vivifytier2", (0, 100), default=3)
    vivifytier3 = Integer("vivifytier3", (0, 100), default=1)
    return [vivifyfocusedtiers, vivifyirr, vivifysort, vivifytier1, vivifytier2, vivifytier3]

# No secondary opt for warmup

def get_options(config_dict) -> ConfigurationSpace:
    options = ConfigurationSpace(seed=0)
    if config_dict.get("backbone", 0) != 0:
        options.add(getBackboneOpt())
    if config_dict.get("bump", 0) != 0:
        options.add(getBumpOpt())
    if config_dict.get("chrono", 0) != 0:
        options.add(getChronoOpt())
    if config_dict.get("congruence", 0) != 0:
        options.add(getCongruenceOpt())
    if config_dict.get("eliminate", 0) != 0:
        options.add(getEliminateOpt())
    if config_dict.get("extract", 0) != 0:
        options.add(getExtractOpt())
    if config_dict.get("factor", 0) != 0:
        options.add(getFactorOpt())
    if config_dict.get("fastel", 0) != 0:
        options.add(getFastelOpt())
    if config_dict.get("forward", 0) != 0:
        options.add(getForwardOpt())
    if config_dict.get("lucky", 0) != 0:
        options.add(getLuckyOpt())
    if config_dict.get("preprocess", 0) != 0:
        options.add(getPreprocessOpt())
    if config_dict.get("probe", 0) != 0:
        options.add(getProbeOpt())
    if config_dict.get("randec", 0) != 0:
        options.add(getRandecOpt())
    if config_dict.get("reluctant", 0) != 0:
        options.add(getReluctantOpt())
    if config_dict.get("reorder", 0) != 0:
        options.add(getReorderOpt())
    if config_dict.get("rephase", 0) != 0:
        options.add(getRephaseOpt())
    if config_dict.get("restart", 0) != 0:
        options.add(getRestartOpt())
    if config_dict.get("stable", 0) != 0:
        options.add(getStableOpt())
    if config_dict.get("substitute", 0) != 0:
        options.add(getSubstituteOpt())
    if config_dict.get("sweep", 0) != 0:
        options.add(getSweepOpt())
    if config_dict.get("transitive", 0) != 0:
        options.add(getTransitiveOpt())
    if config_dict.get("vivify", 0) != 0:
        options.add(getVivifyOpt())
    return options

def getall() -> ConfigurationSpace:
    options = ConfigurationSpace(seed=0)
    options.add(getBackboneOpt())
    options.add(getBumpOpt())
    options.add(getChronoOpt())
    options.add(getCongruenceOpt())
    options.add(getEliminateOpt())
    options.add(getExtractOpt())
    options.add(getFactorOpt())
    options.add(getFastelOpt())
    options.add(getForwardOpt())
    options.add(getLuckyOpt())
    options.add(getPreprocessOpt())
    options.add(getProbeOpt())
    options.add(getRandecOpt())
    options.add(getReluctantOpt())
    options.add(getReorderOpt())
    options.add(getRephaseOpt())
    options.add(getRestartOpt())
    options.add(getStableOpt())
    options.add(getSubstituteOpt())
    options.add(getSweepOpt())
    options.add(getTransitiveOpt())
    options.add(getVivifyOpt())
    return options