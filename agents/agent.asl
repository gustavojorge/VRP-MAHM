agent agent1 {

    belief iteration(0).
    belief max_iterations(100).

    +!start <-
        .print(">> Agente iniciado: ", self),
        !initialize,
        !iterate.

    +!initialize <-
        .py_call("agents.main", "initialize_agent", self).

    +!iterate
        : iteration(I) & max_iterations(M) & I < M <-
        .print(">> Iteração ", I, " do agente ", self),
        .py_call("agents.main", "run_cycle", self),
        I2 = I + 1,
        -iteration(I),
        +iteration(I2),
        !iterate.

    +!iterate
        : iteration(I) & max_iterations(M) & I >= M <-
        .print(">> Agente finalizado: ", self).
}
