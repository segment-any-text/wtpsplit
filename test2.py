from wtpsplit import SaT

sat = SaT(
    "segment-any-text/sat-3l-sm",
    triton_url="localhost:8001",
    triton_model_name="sat_3l_sm"
)
result =  sat.split("This is a test. This is another sentence.")
print(result)