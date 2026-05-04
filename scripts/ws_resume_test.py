"""
Simple WebSocket test client to simulate presentation -> pause -> send chat while paused -> resume.
Usage: python scripts/ws_resume_test.py --url ws://localhost:8000/ws/<session_id>
Requires: websockets (pip install websockets)
"""
import asyncio
import json
import argparse

import websockets

async def run(url):
    async with websockets.connect(url) as ws:
        session_id = url.rstrip('/').split('/')[-1]
        token = 'test-token'
        await ws.send(json.dumps({"type": "start_session", "token": token, "student_id": "tester", "language": "fr"}))
        print('start_session sent')
        # Start presentation
        await ws.send(json.dumps({"type": "start_presentation", "course_id": "c1", "chapter_index": 0, "section_index": 0}))
        print('start_presentation sent')
        # receive some messages for a few seconds
        async def reader():
            try:
                while True:
                    msg = await ws.recv()
                    print('RECV>', msg[:400])
            except Exception as e:
                print('reader ended', e)
        reader_task = asyncio.create_task(reader())
        await asyncio.sleep(4)
        # send a pause with playback snapshot at ~2s
        snapshot = {"kind": "presentation", "currentTime": 2.0, "duration": 10.0}
        await ws.send(json.dumps({"type": "pause", "playback_snapshot": snapshot, "reason": "user_request"}))
        print('pause sent')
        await asyncio.sleep(1)
        # while paused, send a text question
        await ws.send(json.dumps({"type": "text_question", "content": "Peux-tu répéter la dernière partie?", "language": "fr"}))
        print('text_question sent while paused')
        await asyncio.sleep(2)
        # send playback progress update (simulate client reporting current time)
        await ws.send(json.dumps({"type": "playback_update", "playback_snapshot": {"currentTime": 2.1, "duration": 10.0}}))
        print('playback_update sent')
        await asyncio.sleep(1)
        # resume
        await ws.send(json.dumps({"type": "resume"}))
        print('resume sent')
        # let it run a bit
        await asyncio.sleep(6)
        reader_task.cancel()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='ws://localhost:8000/ws/testsession', help='WebSocket URL')
    args = parser.parse_args()
    asyncio.run(run(args.url))
