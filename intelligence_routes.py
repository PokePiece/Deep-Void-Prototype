from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

intelligence_router = APIRouter()
#get the fastAPI router here and import it in core_intelligence

class DiscussionInput(BaseModel):
    prompt: str

@intelligence_router.post('/intelligence')
def intelligence_obtain(input: DiscussionInput):
    from core_intelligence import action, reason, chat
    print('obtaining intelligence')
    user_input_string = input.prompt
    ai_response_content = action(user_input_string)
    return JSONResponse(content={"response": ai_response_content})

@intelligence_router.get("/okcheck")
async def ok_check():
    return JSONResponse(content={"ok": True})