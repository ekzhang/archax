from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.service import hlo_pb2

import jax
import jax.numpy as jnp
import jaxlib.xla_extension as xla


def tanh(x):
    y = jnp.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)


def lfn(x):
    return jnp.log(tanh(x).sum())


def dlfn(x):
    return jax.grad(lfn)(x)


client: xla.Client = jax.lib.xla_bridge.get_backend()


def get_optimized_hlo(c: xla.XlaComputation) -> xla.HloModule:
    e: xla.Executable = client.compile(c)
    modules = e.hlo_modules()
    assert len(modules) == 1, "Expected exactly one HLO module"
    return modules[0]


def variable_shape(shape: xla_data_pb2.ShapeProto) -> str:
    if shape.tuple_shapes:
        return "(" + ", ".join(variable_shape(ty) for ty in shape.tuple_shapes) + ")"
    result = xla_data_pb2.PrimitiveType.Name(shape.element_type)
    if shape.dimensions:
        result += "[" + ",".join(str(size) for size in shape.dimensions) + "]"
    return result


def program_shape(shape: xla_data_pb2.ProgramShapeProto) -> str:
    assert len(shape.parameters) == len(shape.parameter_names)
    params = ", ".join(
        name + ": " + variable_shape(ty)
        for name, ty in zip(shape.parameter_names, shape.parameters)
    )
    ret_ty = variable_shape(shape.result)
    return f"({params}) -> {ret_ty}"


def print_module(module_obj: xla.HloModule) -> None:
    print("\n::: ---------- HLO MODULE ----------")
    print(module_obj.to_string(xla.HloPrintOptions.short_parsable()))


def inspect_module(module_obj: xla.HloModule) -> None:
    print("\n::: ---------- HLO INSPECT ----------")
    module = hlo_pb2.HloModuleProto.FromString(
        module_obj.as_serialized_hlo_module_proto()
    )
    print("name:", module.name)
    print("num computations:", len(module.computations))

    for i, computation in enumerate(module.computations):
        print(f"computation #{i+1}:")
        print(f"  - name: {computation.name}")
        print(f"  - shape: {program_shape(computation.program_shape)}")
        print(f"  - instructions: {len(computation.instructions)}")

    print("cost analysis:", xla.hlo_module_cost_analysis(client, module_obj))


def save_dot_graph(module_obj: xla.HloModule, filename: str) -> None:
    graph = xla.hlo_module_to_dot_graph(module_obj)
    with open(filename, "w") as f:
        f.write(graph)


def main():
    z: xla.XlaComputation = jax.xla_computation(dlfn)(jnp.ones(256))

    original_hlo = z.as_hlo_module()
    print_module(original_hlo)
    inspect_module(original_hlo)
    save_dot_graph(original_hlo, "original_hlo.dot")

    optimized_hlo = get_optimized_hlo(z)
    print_module(optimized_hlo)
    inspect_module(optimized_hlo)
    save_dot_graph(optimized_hlo, "optimized_hlo.dot")


if __name__ == "__main__":
    main()
