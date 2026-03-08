"""Quick test script to debug healthie.find_patient and create_appointment.

Run with:
    uv run test_healthie.py
"""

import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)

import healthie


async def main():
    print("\n--- Testing find_patient ---")
    result = await healthie.find_patient("Rahul Harikumar", "May 17, 2001")
    print(f"find_patient result: {result}")

    if result:
        patient_id = result["patient_id"]
        print(f"\n--- Testing create_appointment for patient {patient_id} ---")
        appt = await healthie.create_appointment(patient_id, "March 10, 2026", "2:00 PM")
        print(f"create_appointment result: {appt}")
    else:
        print("Skipping create_appointment since patient not found.")

    input("\nPress Enter to close the browser...")
    if healthie._browser:
        await healthie._browser.close()


asyncio.run(main())
