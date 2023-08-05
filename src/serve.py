from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
def health_check():
    return {"message": "Server is up and running"}

if __name__ == "__main__":
    uvicorn.run("serve:app",reload=True)