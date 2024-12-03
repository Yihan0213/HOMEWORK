
using Roots,LinearAlgebra,Plots,Optim

#PROBLEM 1
function iterative_solver(f, x0, α, ϵ=1e-6, maxiter=1000)
    x = x0
    residuals = Float64[]
    solutions = Float64[]
    
    for n in 1:maxiter
        g = f(x) + x
        x_new = (1 - α) * g + α * x
        residual = abs(x_new - x)
        push!(residuals, residual)
        push!(solutions, x_new)
        if residual < ϵ
            return 0, x_new, f(x_new), residual, solutions, residuals
        end
        x = x_new
    end
    return 1, NaN, NaN, NaN, solutions, residuals
end

f(x) = (x + 1)^3 - x
x0 = 1
α = 0.0

status, solution, f_value, residual, solutions, residuals = iterative_solver(f, x0, α)


println("Status: ", status)
println("Solution: ", solution)
println("f(solution): ", f_value)
println("Residual: ", residual)

#PROBLEM 2
using LinearAlgebra

function exact_solution(α, β)
    x4 = 1 / (1 - β)
    x3 = β * x4
    x2 = α * x3
    x1 = α * x2
    return [x1, x2, x3, x4, 1]
end


function linear_solver(α, β)
    
    A = [
        1   -1   0   α - β β;
        0    1  -1    0    0;
        0    0   1   -1    0;
        0    0   0    1   -1;
        0    0   0    0    1   
    ]
    
    b = [α, 0, 0, 0, 1]
    x = A \ b
    residual = norm(A * x - b) / norm(b)
    cond_A = cond(A)

    return x, residual, cond_A
end

function create_table(α, β_values)
    println("β | Exact Solution | Backslash Solution | Residual | Condition Number")
    for β in β_values
        exact = exact_solution(α, β)
        x, residual, cond_A = linear_solver(α, β)
        println("$β | $exact | $x | $residual | $cond_A")
    end
end

create_table(0.1, [1, 10, 100, 1000, 10000, 100000])

#Problem 3
using Roots
function NPV(r, C)
    npv = 0.0
    T = length(C)
    for t in 1:T
        npv += C[t] / (1 + r)^(t-1)
    end
    return npv
end

function wrapped_NPV(C)
    return r -> NPV(r, C)
end

function internal_rate(C)
    if all(x -> x >= 0, C) || all(x -> x <= 0, C)
        return"Warning: Cash flow sequence has no sign change, IRR cannot be calculated."
    end
    

    try
        root = find_zero(wrapped_NPV(C), -5.0, 5.0, method = :bisect)
        return root
    catch e
        return
    end
end

cash_flows = [-5, 0, 0, 2.5, 5]
irr = internal_rate(cash_flows)
println("The Internal Rate of Return (IRR) is: ", irr)


#PROBLEN 4
using JuMP
using Ipopt
using Plots

function production_function(x1, x2, alpha, sigma)
    return (alpha * x1^(sigma - 1) / (sigma - 1)) + (1 - alpha) * x2^sigma
end

function cost_function(alpha, sigma, w1, w2, y)
    model = Model(Ipopt.Optimizer)
    @variable(model, x1 >= 0)
    @variable(model, x2 >= 0)
    @constraint(model, production_function(x1, x2, alpha, sigma) == y)
    @objective(model, Min, w1 * x1 + w2 * x2)

    optimize!(model)
    
    cost = objective_value(model)
    optimal_x1 = value(x1)
    optimal_x2 = value(x2)
    
    return cost, optimal_x1, optimal_x2
end

#test
alpha=0.5
w1=1
w2=1
y=1 

sigma_values = [0.25, 1, 4]

costs = Float64[]
x1_values = Float64[]
x2_values = Float64[]

for sigma in sigma_values
    cost, optimal_x1, optimal_x2 = cost_function(alpha, sigma, w1, w2, y)
    push!(costs, cost)
    push!(x1_values, optimal_x1)
    push!(x2_values, optimal_x2)
end


p1 = plot(sigma_values, costs, label="Cost", xlabel="Sigma", ylabel="Cost", lw=2)
p2 = plot(sigma_values, x1_values, label="x1 Demand", xlabel="Sigma", ylabel="x1", lw=2)
p3 = plot(sigma_values, x2_values, label="x2 Demand", xlabel="Sigma", ylabel="x2", lw=2)

plot(p1, p2, p3, layout=(3, 1), size=(800, 600))





