import perceval as pcvl
from perceval.algorithm import Sampler

print("Perceval version:", pcvl.__version__)

# Build circuit and processor
circuit = pcvl.Circuit(2) // pcvl.BS()
processor = pcvl.Processor("SLOS", circuit)
processor.with_input(pcvl.BasicState([1, 0]))

# Get probabilities
sampler = Sampler(processor)
probs = sampler.probs()["results"]

print("P(1,0) =", probs[pcvl.BasicState([1, 0])])
print("P(0,1) =", probs[pcvl.BasicState([0, 1])])
print("Sum    =", probs[pcvl.BasicState([1, 0])] + probs[pcvl.BasicState([0, 1])])