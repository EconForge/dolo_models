path = Pkg.dir("Dolo")

import Dolo


########
######## RBC with a Markov shock
########

# fn = joinpath(path,"examples","models","LAMP.yaml")
model_mc = Dolo.yaml_import("rbc_mc.yaml")

drc = Dolo.ConstantDecisionRule(model_mc.calibration[:controls])
# @time dr0, drv0 = Dolo.solve_policy(model_mc, drc) #, verbose=true, maxit=10000 )
@time dr = Dolo.time_iteration(model_mc, verbose=false, maxit=10000, details=false)
@time drv = Dolo.evaluate_policy(model_mc, dr, verbose=false) #, verbose=true, maxit=10000)
@time drd = Dolo.time_iteration_direct(model_mc, dr, verbose=false, details=false) #, maxit=500)

sim = Dolo.simulate(model_mc, dr) #; N=100, T=20)

########
######## RBC with an iid shock
########

model = Dolo.Model("rbc_iid.yaml")

@time dr = Dolo.perturbate(model)

drc = Dolo.ConstantDecisionRule(model.calibration[:controls])

@time dr = Dolo.time_iteration(model, maxit=100, verbose=false, details=false)
@time res = Dolo.time_iteration(model, dr; maxit=100, details=true)
#
# @time dr0, drv0 = Dolo.solve_policy(model, drc) #;, verbose=true, maxit=1000 )
# @time res = Dolo.solve_policy(model, drc; details=true) #;, verbose=true, maxit=1000 )

@time drd = Dolo.time_iteration_direct(model, details=false) #, maxit=1000, verbose=true)

@time dr = Dolo.time_iteration_direct(model, drd, details=false) #, maxit=500, verbose=true)
@time res = Dolo.time_iteration_direct(model, drc; details=true)
@time drv = Dolo.evaluate_policy(model, dr; verbose=false)

Dolo.simulate(model, dr)

s0 = model.calibration[:states]+0.1
sim = Dolo.simulate(model, dr, s0)
Dolo.simulate(model, dr; N=10)

res = Dolo.response(model.exogenous, [0.01])

irf = Dolo.response(model, dr, :e_z)
irf[:k]

########
######## RBC with an AR1 driving process
########

# AR1 model: this one should be exactly equivalent to rbc_ar1
model = Dolo.Model("rbc_ar1.yaml")
dp = Dolo.discretize(model.exogenous)

@time drp = Dolo.perturbate(model)

cdr =Dolo.CachedDecisionRule(dr,dp)
@time dr = Dolo.time_iteration(model, cdr, details=false)

@time dr = Dolo.time_iteration(model, details=false)
@time drv = Dolo.evaluate_policy(model, dr, verbose=false, maxit=10000)
@time drd = Dolo.time_iteration_direct(model, dr, details=false) #, verbose=true) #, maxit=500)
#
Dolo.simulate(model, dr, N=10)

Dolo.response(model.exogenous, [0.01])

irf = Dolo.response(model, dr, :z)

irf[:z]
