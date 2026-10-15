# Quantum Bias Lab

A lab for watching whether random bits drawn from quantum circuits are biased, and whether a person
can influence or guess them. This file is the glossary; `docs/DESIGN.md` is the contract.

## Language

### Where bits come from

**Provider**:
The kind of machine that measures the bits: `sim` (local simulator) or `ibm` (a real IBM QPU).
_Avoid_: backend, source, vendor

**Variant**:
What a provider is, in words a person reads: simulator or real QPU.
_Avoid_: type, kind

**Implementation**:
The specific device that measured, e.g. `ibm_torino` or `aer_ideal`.
_Avoid_: backend (the API field keeps that name), device type

**Measurement**:
The bits one job produced: one row per shot, one column per logical qubit, with the job and device
that produced them.
_Avoid_: result, sample

**Pool**:
Measurements taken earlier for one provider, implementation and qubit count, handed out in order
one shot at a time and never reused.
_Avoid_: buffer, cache, reservoir

**Pool feeder**:
Everything that puts bits into a pool: an explicit prefetch, an auto refill, and the bits of a
discarded live job.
_Avoid_: refiller, loader

**Prefetch**:
A job the user starts on purpose to fill a pool.
_Avoid_: preload, warm-up

**Auto refill**:
A prefetch a session starts by itself when its pool runs low; simulator only, and never retried
after it fails.
_Avoid_: top-up (the code's verb, not the concept), auto-prefetch

### Getting bits for a person

**Acquisition mode**:
How a session gets its bits: **live** (a fresh job measured after the request) or **buffered**
(the next shots from a pool, measured before the request).
_Avoid_: source mode, fetch mode

**Acquisition**:
Getting the bits for one interaction or one stream cycle, in the acquisition mode of the segment
it started in.
_Avoid_: fetch, draw, read

**Session**:
One person's run of the lab: one provider, qubit count and aggregation, with its statistics.
_Avoid_: experiment, game

**Segment**:
A stretch of a session in one acquisition mode; switching mode starts a new one, and statistics
are kept per segment.
_Avoid_: phase, period

**Provenance**:
What every bit carries about its origin: job, provider, device, acquisition mode and when it was
measured. Buffered bits never pass as live ones.
_Avoid_: metadata, origin info

**Aggregation**:
How one bit is made from the qubits of one shot: qubit 0 alone, the XOR of all, or the majority.
_Avoid_: combining, reduction

### Interactions

**Key press**:
One aggregated bit on request.

**Predict**:
The person guesses first, then the bit is acquired.

**Perceive**:
A bit is acquired and hidden, the person guesses, then it is revealed.

**Stream**:
Bits delivered continuously, one cycle after another, until stopped.

**Stream run**:
One start-to-stop of a stream, bounded by a duration and, on `ibm`, by a job count.
_Avoid_: stream session, streaming job

**Cycle**:
One step of a stream run: one acquisition and the batch of bits it delivers.
_Avoid_: tick, iteration, round

**Permit**:
The stream run's yes to one more job, checked and counted at the moment the job is submitted.
_Avoid_: quota, token, slot
