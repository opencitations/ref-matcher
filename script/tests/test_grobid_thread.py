"""The GROBID call is blocking HTTP: it must run in a thread, so the other references in
flight keep querying meanwhile (before, the whole event loop stopped for each call)."""
import asyncio
import time


def test_grobid_does_not_stop_other_references(rmt):
    T = rmt

    class GrobidLento:
        def process_unstructured_reference(self, text):
            time.sleep(0.5)                              # a slow GROBID answer
            return None

    P = T.ReferenceProcessor(use_grobid=False)
    P.use_grobid, P._grobid_instance = True, GrobidLento()     # what the grobid_processor property returns
    passi = []

    async def altro_lavoro():                            # another reference querying meanwhile
        for _ in range(10):
            passi.append(time.perf_counter())
            await asyncio.sleep(0.02)

    class Muto(T.OpenCitationsMatcherThreadSafe):
        async def query_opencitations(self, *a, **k):
            return []

    async def go():
        ref = T.Reference(unstructured='Barber B (2000) Trading is hazardous. J Finance 55:773-806')
        async with Muto(query_cache=None) as M:
            await asyncio.gather(P.process_reference(ref, M, 26, False), altro_lavoro())

    t0 = time.perf_counter()
    asyncio.run(go())
    assert passi[-1] - t0 < 0.45                         # finished while GROBID was still busy
