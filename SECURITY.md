# Security — in plain language

This document explains, without assuming any cybersecurity background, the
safety measures built into Outbreak and why they exist. If you just want to run
the simulator on your own computer, you don't need to do anything — these
protections work automatically. This is here so you understand *what* they do
and *why*.

---

## The one-paragraph version

Outbreak lets you **save a simulation to a file** and **load it back later**. A
file like that could, in principle, come from someone else — and a file from a
stranger should never be trusted blindly, the same way you wouldn't run a
program emailed to you by someone you don't know. We added a few simple checks
so that opening a saved file can, at worst, show a polite error message — it can
never crash your computer, eat all its memory, or make the app misbehave. We
also "locked" the list of helper software the project uses, so everyone runs the
exact same, verified versions.

---

## Background: what is a "snapshot"?

While a simulation runs, the program holds a lot of information in memory: how
many people are healthy, sick, recovered, and so on. A **snapshot** is simply
all of that information written out to a file (in a common text format called
JSON). You can save a snapshot and re-open it later to carry on exactly where
you left off, or send it to a colleague.

That convenience is also the thing we have to be careful about: **a file can
contain anything.** Most of the time it's a genuine snapshot you saved earlier.
But the program can't assume that. Someone could hand you a file that *looks*
like a snapshot but has been deliberately altered to cause trouble.

## Why a "bad" file is a risk

Think of the program reading a snapshot like a builder following a set of
instructions. A normal snapshot says something reasonable like "there are four
age groups, here are their numbers." A malicious snapshot might instead say:

- **"There are 50 billion age groups."** A naïve program would dutifully try to
  set aside enough memory for all of them and grind the computer to a halt. This
  is called a *denial-of-service* — not someone stealing anything, just making
  the tool unusable by exhausting its resources.
- **"Here is a number where I promised a list,"** or **"this list is a different
  length than that one."** A naïve program might get confused halfway through and
  fall over with a frightening, technical error — or worse, keep going in a
  broken state.

None of this could let an attacker take over your machine (the program never
treats a file as commands to run). The realistic worst cases were "the app
crashes" or "the app freezes," and we've now closed both off.

## What we changed (the three safeguards)

### 1. A size limit checked *before* the file is opened

The app now looks at how big an uploaded file is **before** it tries to read it.
Anything larger than a generous limit (50 megabytes — far bigger than any real
snapshot) is refused immediately with a clear message. This is the key one: it
stops the "50 billion age groups" trick, because the oversized file never gets
read in the first place.

> Analogy: checking the *weight* of a parcel before opening it, rather than
> opening it and discovering too late that it's full of bricks.

### 2. Every piece of a snapshot is checked for the right "shape"

When the program loads the contents of a snapshot, it now verifies that each
part is exactly what it expects: the right *kind* of value, the right *size*,
and within sensible *limits*. For example, an "age group" number has to point to
a real age group; a person's health status has to be one of the statuses the
model actually uses. If anything doesn't fit, the program stops politely and
explains what was wrong, instead of pushing ahead with nonsense.

> Analogy: a jigsaw piece that's the wrong shape simply won't be forced into the
> slot — you're told it doesn't fit.

This applies to every part of a snapshot, including the contact-network data
(which household/class/workplace each simulated person belongs to): those values
are range-checked too, so a tampered file can't point a person at a group that
doesn't exist.

### 3. Friendly errors instead of scary crashes

If a file is corrupted, incomplete, or just isn't a real snapshot, the app now
catches the problem and shows a short message like *"Could not load snapshot."*
Previously it could spill out a wall of technical text. Now it fails gently and
the app keeps running.

## Locking the helper software (reproducibility & supply chain)

Outbreak relies on well-known, free building-block libraries (for number
crunching, charts, and the web interface). Previously the project asked for
"version X *or newer*" of each, which means two people installing it on different
days could quietly end up with different versions.

We added a **lockfile** (`requirements.lock`) that pins **every** library to one
exact, known-good version and records a unique fingerprint (a "hash") for each.
When you install from it, the installer verifies every download against its
recorded fingerprint and refuses anything that doesn't match.

> Analogy: instead of "buy some milk," the shopping list now says "buy this
> exact carton," and the checkout double-checks the barcode. If a library were
> ever tampered with upstream, the fingerprint wouldn't match and installation
> would stop.

To install the locked, verified set:

```bash
pip install --require-hashes -r requirements.lock
```

## If you put Outbreak on the internet

Outbreak is designed to run **on your own computer, for one person at a time**.
It has no login system and no user accounts — it doesn't need them for local
use.

If you ever host it on a public web address so that strangers can reach it,
please put it behind something that handles **logins/access control** and
**limits how much work one visitor can request** (running a huge simulation
takes real computing power). The safeguards above make the *file-loading* part
safe to expose, but the app as a whole assumes a trusted, single-user setting.

## Reporting a concern

If you believe you've found a security problem, please open an issue describing
what you saw and how to reproduce it. There is no sensitive data in this
project, so normal issue reporting is fine.
