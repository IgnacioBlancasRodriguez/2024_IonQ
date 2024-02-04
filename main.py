from typing import List
from qiskit import *
from qiskit.quantum_info import Statevector, partial_trace
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import numpy as np
import math

class Circuit(BaseModel):
    name: str
    qr : int
    cr: int

class CircuitStates(BaseModel):
    name: str
    status: str
    states: List[List[float]]

class GateOp(BaseModel):
    qbit: int
    op: str
    ang: float

class MultiGateOp(BaseModel):
    qbits: List[int]
    op: str

class CircuitIdentifier(BaseModel):
    name: str

class ExperimentConditions(BaseModel):
    num_shots: int

app = FastAPI()

# Constants
circuit = None
circuit_name = ""
QR = 0
CR = 0
state_vector_backend = Aer.get_backend("statevector_simulator")
stats_backend = Aer.get_backend("qasm_simulator")

def get_statevec_coords(statevector):
    # Convert to polar form:
    r_0 = np.abs(statevector[0])
    phi_0 = np.angle(statevector[0])

    r_1 = np.abs(statevector[1])
    phi_1 = np.angle(statevector[1])

    # Calculate the coordinates:
    r = np.sqrt(r_0 ** 2 + r_1 ** 2)
    theta = 2 * np.arccos(r_0 / r)
    phi = phi_1 - phi_0
    z = r * math.cos(theta)
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    return [x, y, z]

@app.post("/generate_circuit/")
def generate_circuit(circ: Circuit):
    global circuit, circuit_name, QR, CR

    circuit_name = circ.name
    QR = circ.qr
    CR = circ.cr
    circuit = QuantumCircuit(circ.qr, circ.cr)
    return JSONResponse(status_code=200, content={"msg": "Succesfully created"})

@app.post("/apply_multi_gate")
def apply_multi_gate(multi_gate_op: MultiGateOp):
    global circuit
    for qbit in multi_gate_op.qbits:
        if qbit >= QR:
            return JSONResponse(status_code=400, content=
                {"msg": "Requested Qbits are not within the circuit"})
    if multi_gate_op.op == "CNOT":
        circuit.cnot(multi_gate_op.qbits[0], multi_gate_op.qbits[1])
        return JSONResponse(status_code=200, content=
            {"msg": "Succesfully applied cnot gate on control {} and target {} qbits".format(
                multi_gate_op.qbits[0], multi_gate_op.qbits[1])})
    if multi_gate_op.op == "CZ":
        circuit.cz(multi_gate_op.qbits[0], multi_gate_op.qbits[1])
        return JSONResponse(status_code=200, content=
            {"msg": "Succesfully applied cz gate on control {} and target {} qbits".format(
                multi_gate_op.qbits[0], multi_gate_op.qbits[1])})

@app.post("/apply_single_gate")
def apply_single_gate(gate_op: GateOp):
    global circuit

    if gate_op.qbit >= QR:
        return JSONResponse(status_code=400,
            content={"msg": "Requested Qbit is not within the circuit"})
    if gate_op.op == "HGate":
        circuit.h(gate_op.qbit)
        return JSONResponse(status_code=200, content=
            {"msg": "Succesfully applied a hadamard gate on qbit {}".format(gate_op.qbit)})
    elif gate_op.op == "PXGate":
        circuit.x(gate_op.qbit)
        return JSONResponse(status_code=200, content=
            {"msg": "Succesfully applied a X gate on qbit {}".format(gate_op.qbit)})
    elif gate_op.op == "PYGate":
        circuit.y(gate_op.qbit)
        return JSONResponse(status_code=200, content=
            {"msg": "Succesfully applied a Y gate on qbit {}".format(gate_op.qbit)})
    elif gate_op.op == "PZGate":
        circuit.z(gate_op.qbit)
        return JSONResponse(status_code=200, content=
            {"msg": "Succesfully applied a Z gate on qbit {}".format(gate_op.qbit)})
    elif gate_op.op == "SGate":
        circuit.s(gate_op.qbit)
        return JSONResponse(status_code=200, content=
            {"msg": "Succesfully applied a S gate on qbit {}".format(gate_op.qbit)})
    elif gate_op.op == "TGate":
        circuit.t(gate_op.qbit)
        return JSONResponse(status_code=200, content=
            {"msg": "Succesfully applied a T gate on qbit {}".format(gate_op.qbit)})
    elif gate_op.op == "Rx":
        if gate_op.ang < -math.pi or gate_op.ang > math.pi:
            return JSONResponse(status_code=400, content=
                {"msg": "Provided angle is outside of the available bounds"})
        circuit.rx(gate_op.ang, gate_op.qbit)
        return JSONResponse(status_code=200, content=
            {"msg": "Succesfully applied a Rx gate on qbit {}".format(gate_op.qbit)})

@app.post("/measure_all")
def measure_all(circ_id: CircuitIdentifier):
    if circ_id.name != circuit_name:
        return JSONResponse(status_code=403, content=
            {"msg": "Given name is not the name of the current circuit"})
    circuit.measure_all()
    return JSONResponse(status_code=200, content=
        {"msg": "Succesfully applied the measurement gates"})

@app.get("/measure_all_once")
def measure_all_once():
    if circuit == None:
        return JSONResponse(status_code=400, content=
            {"msg": "No circuit"})
    circuit.measure_all()
    resulting_cnts = execute(circuit, stats_backend, shots=1).result().get_counts()
    return list(resulting_cnts.keys())[0]

@app.get("/get_results")
def get_results(expCond: ExperimentConditions):
    if expCond.num_shots <= 0:
        return JSONResponse(status_code=400, content=
            {"msg": "number of shots provided is too small"})
    resulting_cnts = execute(circuit, stats_backend, shots=expCond.num_shots).result().get_counts()
    for key in resulting_cnts.keys():
        resulting_cnts[key] /= expCond.num_shots

    return JSONResponse(status_code=200, content=resulting_cnts)

@app.get("/get_state_vectors")
def get_state_vectors():
    if circuit == None:
        return JSONResponse(content=
            {"name":circuit_name, "status": "No circuit", "states": []})
    qbit_vecs = ""
    state = execute(circuit, state_vector_backend).result().get_statevector()
    for i in range(0, state.num_qubits):
        partial_density_matrix = partial_trace(state,
            list(range(0, i)) + list(range(i + 1, state.num_qubits)))
        partial_statevector = Statevector(np.diagonal(partial_density_matrix))
        raw_vec = get_statevec_coords(partial_statevector)
        raw_vec = [0 if abs(coord) <= 0.00001 else coord for coord in raw_vec]
        for coord in raw_vec:
            qbit_vecs += "," + str(coord)
    return qbit_vecs
