---
title: "It Worked in Dev. It Worked in QA. Then Production Happened."
description: "A backend production incident where an appointment endpoint caused 4-second response times due to an N+1 query problem with 6,000+ DB calls, and how we solved it with query batching and projections."
date: 2026-01-11
category: "Architecture"
tags: ["database", "n+1-query", "performance", "backend", "mongodb"]
draft: false
readingTime: 5
---

It’s interesting how we approach problems at different stages of the software development life cycle.

A few months back at Medoc (where I am currently working as a backend engineer, and since it’s a growing startup, a lot of things are being handled by me), I asked my teammate (a fresher) to write an endpoint for fetching today’s appointments and those that are not completed for a doctor.

It was quite simple because we have a separate collection for appointments with an `eventDate` field, so he just needed to query it based on the current time. But here was the catch: the frontend also required patient details along with the appointments. The patient data is stored in a separate database.

So it became a case of dual database calls, where the first call fetches the appointments and the second call fetches the associated patient details, which is a typical **N+1 query problem**.

---

## The Initial Implementation

What my teammate did, like most people would, was:

1. `fetch appointments`
2. `loop over each appointment`
3. `fetch patient details one by one`

And it worked.

Manual testing passed, it was deployed to dev for QA testing, and that passed too.

Soon it was deployed to production and worked fine for over two months. Everything seemed sorted.

---

## Production Happened

Then a few days back, I got a complaint from the frontend team that their home screen was getting stuck, and that too for only one doctor, which was strange. A frontend friend and I started debugging it, but here we made a mistake. We were testing in the dev environment and were not able to reproduce the issue.

We spent almost an hour trying to fix and optimize the UI. Our initial thought was that since it was a home screen, it was hitting too many endpoints while loading the initial data. But we couldn’t find the real culprit.

Then we checked the production data for that doctor. Once we accessed the user account and opened the appointments view, we found the issue.

This particular doctor had **6,000+ appointments**, which is a lot for a startup, and he had marked only a very small number as completed, so the incoming data was huge.

---

## Pinpointing the Culprit

I immediately started pinpointing the API responsible for fetching these appointments. As soon as I looked at its implementation, the problem became clear.

It was making **6,000+ database calls** to retrieve patient data. That’s why the response time was around **4 seconds**, which was never caught during initial testing because we never tested with this much data.

---

## The Refactoring & Optimization

I immediately refactored the code using in-memory maps (feel free to comment if you have a better approach).

The implementation was simple:

1. `first, fetch all required appointments`
2. `then create an array of patient IDs`
3. `fetch all patient details in a single database call`
4. `create a map with patient ID as the key and patient details as the value`
5. `loop over the appointments again to construct the DTO`

I also optimized a few things. This route was over-fetching data, so I added proper projections and fetched only the required fields. This reduced the payload size as well.

After this, the latency came down to **500–600 ms**, which was a huge difference.

---

## Now Comes the Real Lesson

The fault was not just the developer’s. It was also:

- **me**, for merging the PR without a deep review and not clarifying constraints
- **the tester**, who tested only in a controlled environment
- **the developer**, who didn’t think about edge cases or ask for clarity

But I believe these kinds of unexpected production issues are what make developers grow and face the real horror of production.

> While writing code, always expect the worst and hope for the best.
>
> — _Nikhil_
