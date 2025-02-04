import DiskVec
import time

dv = DiskVec.DiskVec("embeddings.dat")
print(dv.create(100000, 128))
print(dv.load())
# now perform search
start = time.time()
result = dv.search([0.5] * 128, 6)
print(f"Searched in {time.time() - start} secs")
print("Search result:", result)
