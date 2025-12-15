import asyncio
from spade_bdi.bdi import BDIAgent

async def main():
    agent = BDIAgent(
        "agent1@localhost",
        "123",
        "agents/agent.asl"
    )

    await agent.start()
    await asyncio.sleep(5)
    await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
