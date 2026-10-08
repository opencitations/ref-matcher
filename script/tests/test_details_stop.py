"""query_opencitations fetches details in groups: stop() sees only each new group's rows
(rescoring all rows every group was quadratic) and still stops at the same group."""
import asyncio
import re


def test_stop_sees_only_new_rows_and_stops_at_same_group(rmt):
    T = rmt
    cands = [f'http://x/br/{n}' for n in range(100)]
    calls = []

    async def fake(q, qt):
        if qt.endswith(':candidates'):
            return [{'br': {'value': c}} for c in cands]
        return [{'br': {'value': c.strip('<>')}} for c in re.search(r'VALUES \?br \{([^}]*)\}', q)[1].split()]

    def stop(rows):
        calls.append(len(rows))
        return any(r['br']['value'] == cands[50] for r in rows)    # match in the 4th group

    m = T.OpenCitationsMatcherThreadSafe(query_cache=None)
    m._execute_query = fake
    rows = asyncio.run(m.query_opencitations(
        'SELECT * WHERE { ?br ?p ?o . OPTIONAL { ?br <http://x/y> ?z } }', 'author_title', stop=stop))
    assert calls == [T.DETAILS_GROUP] * 4                  # one group each time, not the growing list
    assert len(rows) == 4 * T.DETAILS_GROUP                # stopped after the group holding the match
