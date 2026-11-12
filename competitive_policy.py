"""Damage/cancellation/life-pressure value using only observed conservative data."""


def combat_tempo_value(move, exchange_result, incoming, combat, height):
    """Attack before garbage lands, but refuse death for speculative damage."""
    incoming=max(0,int(incoming))
    if combat=='ATTACK':
        return min(exchange_result.sent,12)*.7
    if combat=='COUNTER':
        return min(exchange_result.cancelled,12)*1.7+min(exchange_result.sent,8)*.4
    if combat=='DEFENSE':
        return min(exchange_result.cancelled,20)*2.3 - max(0,height+exchange_result.remaining-16)*8.0
    if combat=='EMERGENCY':
        return min(exchange_result.cancelled,20)*3.0 - max(0,height+exchange_result.remaining-15)*16.0
    return 0.0
