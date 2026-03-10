from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from schema import UserInput, PredictionResponse
from model import predict_output


app = FastAPI()


@app.get("/")
async def home():
    return {"message": "AI Research Tool"}


@app.get("/health")
async def health_check():
    return {
        "status": "OK",
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",  # Add this line
        "http://localhost:5501",  # Add this line
    ],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.post("/summarize", response_model=PredictionResponse)
def predict_premium(data: UserInput):
    input = {
        # "message": data.message,
        "paper_input": data.paper_input,
        "style_input": data.style_input,
        "length_input": data.length_input,
    }

    try:
        prediction = predict_output(input)
        return JSONResponse(status_code=200, content={"response": prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content=str(e))
