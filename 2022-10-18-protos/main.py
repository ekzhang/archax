from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.service import hlo_pb2

import jax.numpy as jnp
from jax import grad, xla_computation


def tanh(x):
    y = jnp.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)


def lfn(x):
    return jnp.log(tanh(x).sum())


def dlfn(x):
    return grad(lfn)(x)


z = xla_computation(dlfn)(jnp.ones(256))


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


print(z.as_hlo_text())

module = hlo_pb2.HloModuleProto.FromString(z.as_serialized_hlo_module_proto())
print("name:", module.name)
print("num computations:", len(module.computations))

for i, computation in enumerate(module.computations):
    print(f"computation #{i+1}:")
    print(f"  - name: {computation.name}")
    print(f"  - shape: {program_shape(computation.program_shape)}")
    print(f"  - instructions: {len(computation.instructions)}")
