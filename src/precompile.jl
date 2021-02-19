function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(MathOptInterface.optimize!),Optimizer})   # time: 8.211051
    isdefined(Ipopt, Symbol("#eval_jac_g_cb#49")) && Base.precompile(Tuple{getfield(Ipopt, Symbol("#eval_jac_g_cb#49")),Vector{Float64},Symbol,Vector{Int32},Vector{Int32},Vector{Float64}})   # time: 0.046027973
    isdefined(Ipopt, Symbol("#eval_h_cb#50")) && Base.precompile(Tuple{getfield(Ipopt, Symbol("#eval_h_cb#50")),Vector{Float64},Symbol,Vector{Int32},Vector{Int32},Float64,Vector{Float64},Vector{Float64}})   # time: 0.018323332
    isdefined(Ipopt, Symbol("#eval_grad_f_cb#47")) && Base.precompile(Tuple{getfield(Ipopt, Symbol("#eval_grad_f_cb#47")),Vector{Float64},Vector{Float64}})   # time: 0.014712464
    Base.precompile(Tuple{typeof(MathOptInterface.add_variables),Optimizer,Int64})   # time: 0.013180094
    Base.precompile(Tuple{typeof(MathOptInterface.empty!),Optimizer})   # time: 0.008900179
    Base.precompile(Tuple{typeof(MathOptInterface.delete),Optimizer,MathOptInterface.ConstraintIndex{MathOptInterface.SingleVariable, MathOptInterface.GreaterThan{Float64}}})   # time: 0.007912449
    isdefined(Ipopt, Symbol("#eval_g_cb#48")) && Base.precompile(Tuple{getfield(Ipopt, Symbol("#eval_g_cb#48")),Vector{Float64},Vector{Float64}})   # time: 0.007204291
    Base.precompile(Tuple{typeof(MathOptInterface.set),Optimizer,MathOptInterface.ObjectiveFunction,MathOptInterface.ScalarAffineFunction{_A} where _A})   # time: 0.005816489
    Base.precompile(Tuple{typeof(MathOptInterface.features_available),EmptyNLPEvaluator})   # time: 0.00577202
    Base.precompile(Tuple{typeof(MathOptInterface.add_constraint),Optimizer,MathOptInterface.ScalarAffineFunction{Float64},MathOptInterface.EqualTo{Float64}})   # time: 0.00475595
    Base.precompile(Tuple{typeof(MathOptInterface.add_constraint),Optimizer,MathOptInterface.ScalarQuadraticFunction{Float64},MathOptInterface.LessThan{Float64}})   # time: 0.004524159
    Base.precompile(Tuple{typeof(MathOptInterface.get),Optimizer,MathOptInterface.TerminationStatus})   # time: 0.003869384
    Base.precompile(Tuple{typeof(MathOptInterface.set),Optimizer,MathOptInterface.ObjectiveFunction{MathOptInterface.SingleVariable},MathOptInterface.SingleVariable})   # time: 0.002992182
    Base.precompile(Tuple{typeof(MathOptInterface.set),Optimizer,MathOptInterface.Silent,Bool})   # time: 0.002608517
    Base.precompile(Tuple{Type{IpoptProblem},Ptr{Nothing},Int64,Int64,Function,Function,Function,Function,Nothing})   # time: 0.002365634
    Base.precompile(Tuple{typeof(MathOptInterface.set),Optimizer,MathOptInterface.VariablePrimalStart,MathOptInterface.VariableIndex,Union{Nothing, Real}})   # time: 0.002105908
    Base.precompile(Tuple{typeof(freeProblem),IpoptProblem})   # time: 0.001901743
    Base.precompile(Tuple{typeof(MathOptInterface.set),Optimizer,MathOptInterface.ObjectiveFunction,Union{MathOptInterface.SingleVariable, MathOptInterface.ScalarAffineFunction, MathOptInterface.ScalarQuadraticFunction}})   # time: 0.001721906
    Base.precompile(Tuple{Type{IpoptProblem},Ptr{Nothing},Int64,Int64,Function,Function,Function,Function,Function})   # time: 0.001538395
    Base.precompile(Tuple{typeof(MathOptInterface.get),Optimizer,MathOptInterface.ObjectiveValue})   # time: 0.001434465
    Base.precompile(Tuple{typeof(MathOptInterface.get),Optimizer,MathOptInterface.VariablePrimal,MathOptInterface.VariableIndex})   # time: 0.001362476
    Base.precompile(Tuple{typeof(MathOptInterface.get),Optimizer,MathOptInterface.PrimalStatus})   # time: 0.001347455
    isdefined(Ipopt, Symbol("#eval_f_cb#46")) && Base.precompile(Tuple{getfield(Ipopt, Symbol("#eval_f_cb#46")),Vector{Float64}})   # time: 0.001338227
    Base.precompile(Tuple{typeof(MathOptInterface.add_constraint),Optimizer,MathOptInterface.SingleVariable,MathOptInterface.GreaterThan{Float64}})   # time: 0.001213452
    Base.precompile(Tuple{typeof(MathOptInterface.add_constraint),Optimizer,MathOptInterface.SingleVariable,MathOptInterface.LessThan{Float64}})   # time: 0.001103924
    Base.precompile(Tuple{typeof(MathOptInterface.delete),Optimizer,MathOptInterface.ConstraintIndex{MathOptInterface.SingleVariable, MathOptInterface.LessThan{Float64}}})   # time: 0.0010774
end
