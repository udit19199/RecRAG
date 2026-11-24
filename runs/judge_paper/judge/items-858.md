Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 2 items.

=== ITEM id=3e4d811979722683 ===
Gold sentences and facts:
Sentence 0: 13 Dec 2023 00.58 GMT 12th over: West Indies 102-4 (Hope 26, Hetmyer 1) Just a single and the wicket off the over, Shai Hope can’t break the legspin shackles.
  Fact 0.0: The timestamp is 13 Dec 2023, 00:58 GMT.
  Fact 0.1: It is the 12th over.
  Fact 0.2: West Indies were 102 for 4 wickets.
  Fact 0.3: Hope had 26 runs and Hetmyer had 1 run.
  Fact 0.4: In that over there was just a single and a wicket.
  Fact 0.5: Shai Hope could not break the legspin shackles.
Sentence 1: 6d ago 22.54 GMT 16th over: West Indies 112-4 (Hope 29, Powell 8) Hope, who is playing an exemplary innings, opens the face to steer Rehan for his first boundary.
  Fact 1.0: The timestamp is 6 days ago, 22:54 GMT.
  Fact 1.1: It is the 16th over.
  Fact 1.2: West Indies were 112 for 4 wickets.
  Fact 1.3: Hope had 29 runs and Powell had 8 runs.
  Fact 1.4: Hope is playing an exemplary innings.
  Fact 1.5: Hope opens the face to steer Rehan for his first boundary.

Retrieved items:
<item 1>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 23.00 GMT 17th over: West Indies 114-5 (Hope 30, Russell 1) The wicket England really need is that of Shai Hope, who is quietly guiding West Indies
towards the finish line.
</item 1>

<item 2>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Photograph: Ricardo Mazalán/AP 𝗔 𝗦𝗜𝗫 𝗧𝗢 𝗙𝗜𝗡𝗜𝗦𝗛! 🙌@WindiesCricket claim the series 3-2 and a white-ball clean-sweep 🏏#WIvENG pic.twitter.com/chrYkHbtcp — Cricket
on TNT Sports (@cricketontnt) December 21, 2023

Updated at 23.34 GMT

6d ago 23.13 GMT 19.1 overs: West Indies 127-6 (Hope 37, Holder 4) Holder inside edges Woakes this far wide of leg stump for three runs. It would have been four
but for an outstanding stop from Rehan Ahmed.
</item 2>

<item 3>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
12.4 overs: West Indies 108-4 (Hope 27, Powell 6)

Updated at 01.09 GMT

13 Dec 2023 00.58 GMT 12th over: West Indies 102-4 (Hope 26, Hetmyer 1) Just a single and the wicket off the over, Shai Hope can’t break the legspin shackles.
Windies skipper Rovman Powell is the new man and Buttler cranks the pressure by inserting a slip.

Updated at 01.04 GMT

13 Dec 2023 00.52 GMT WICKET! Hetmyer c Duckett b Rashid 1 (West Indies 101-4) Adil Rashid takes his 100th T20I wicket! A crucial one in the context of the game
too – the dangerous Hetmyer is sent on his way for just a single as England look to ramp up the pressure heading into the nub end of this match.
</item 3>

<item 4>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
It would have been four but for an outstanding stop from Rehan Ahmed.



6d ago 23.12 GMT 19th over; West Indies 124-6 (Hope 37, Holder 1) Holder is almost run out first ball! He took a very dodgy single to mid-off and was well short
of the crease when Curran just failed to gather Rashid’s throw. Nine to win from six balls, Chris Woakes to bowl them.



6d ago 23.10 GMT WICKET! West Indies 123-6 (Russell c Livingstone b Curran 3) Oh my. Russell slugs a low full toss straight down the throat of Livingstone at
long-on, and England – who have been behind the game throughout this innings – are again dreaming of an improbable victory. West Indies need 10 from 7 balls.
</item 4>

<item 5>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.33 GMT 11th over: West Indies 71-3 (Hope 14, Rutherford 12) After 23 balls without a West Indian boundary, Rutherford lifts Moeen handsomely over
extra cover for four. That’s an excellent shot which keeps West Indies in control of the run-chase.

Updated at 22.33 GMT

6d ago 22.28 GMT 10th over: West Indies 62-3 (Hope 12, Rutherford 5) West Indies know how dangerous Rashid is and are content to sit on him. I think that was
Charles’ plan as well, but then he saw a low full toss and his brain went rogue. Time for drinks, after which West Indies need 71 from 60 balls.



6d ago 22.25 GMT Rutherford is not out! I need someone else’s glasses: it was comfortably missing leg stump and England have lost a review.
</item 5>

<item 6>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Updated at 23.20 GMT

6d ago 23.16 GMT I suppose it’s appropriate that it ended with a six, the 120th of the series. Hope played a gem of an innings there, calmly anchoring West
Indies in an increasingly nervous chase with a run-a-ball 43. England fought admirably to defend a below-par total; but for Hope, they would probably have done
it.

Updated at 23.17 GMT

6d ago 23.14 GMT WEST INDIES WIN THE SERIES 3-2! 19.2 overs: West Indies 133-6 (Hope 43, Holder 4) That’s it! Hope pings Woakes over extra cover for six to seal
another series victory for West Indies! West Indies’ Shai Hope (right) celebrates with Jason Holder after hitting a six to win the match by 4 wickets.
</item 6>

<item 7>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.37 GMT 12th over: West Indies 80-3 (Hope 15, Rutherford 20) England appeal unsuccessfully for caught behind when Rutherford misses a swipe at Rashid.
There’s only one review left and they decide not to risk it. A good job too; there was nothing on UltraEdge. Out of nothing Rutherford launches a full ball from
Rashid down the ground for six, almost knocking himself off his feet in the process. West Indies are managing an awkward chase pretty well.
</item 7>

<item 8>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 01.45 GMT WICKET! Shai Hope c Brook b Rehan Ahmed 36 (West Indies 123-5) Ahmed is bunted for SIX by Shai Hope but strikes with the next ball as Hope
holes out!



13 Dec 2023 01.44 GMT 14th over: West Indies 112-4 (Hope 27, Powell 10) Liam Livingstone instead of Rashid and he does well to keep it to singles, England
trying to take this as deep as possible and hope West Indies fluff their lines at the death. West Indies need 55 from 36 balls.
</item 8>

<item 9>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
after newsletter promotion

Quick Guide How do I sign up for sport breaking news alerts? Show Download the Guardian app from the iOS App Store on iPhone or the Google Play store on Android
by searching for 'The Guardian'.

If you already have the Guardian app, make sure you’re on the most recent version.

In the Guardian app, tap the Menu button at the bottom right, then go to Settings (the gear icon), then Notifications.

Turn on sport notifications. Was this helpful? Thank you for your feedback.

Halfway through their innings they were 62 for three, and if England had 15 more runs at the same point they had also made a complete hash of the rest of it.
That was a trap which, thanks to Hope’s 43-ball 43 as he again steered his side to their target, West Indies avoided. This batting display may have lacked their
usual flamboyance but until the final ball it was entirely appropriate for the occasion.

Amid the wider theme of transformation Adil Rashid’s excellence remains constant.
</item 9>

<item 10>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.10 GMT 3rd over: West Indies 36-1 (Mayers 10, Hope 4) Shai Hope is the new man and digs out a Woakes yorker and somehow manages to crunch it away
down the ground for four. The replay of the Duckett catch is extremely enjoyable, Woakes owes him one for that as it was a poor ball outside off stump.



13 Dec 2023 00.07 GMT WICKET! Brandon King c Duckett b Woakes 22 (West Indies 32-1) Chris Woakes into the attack to replace the beleaguered Sam Curran. OH. MY.
LORDY. Ben Duckett flies to his left at backward point to pull off a spectacular diving catch off a full blooded cut shot from King. Stunner!
</item 10>

<item 11>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.20 GMT 5th over: West Indies 56-1 (Mayers 20, Hope 12) Woakes continues, after five balls he’s gone for just three runs. Can he get out of the
over cheaply? No – A length ball is clubbed down the ground for SIX with a minimum of fuss by Shai Hope. West Indies have come out of the traps as well as
England did, one over of the Powerplay left.
</item 11>

<item 12>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Adil Rashid (right) celebrates the dismissal of Johnson Charles (left). Photograph: Randy Brooks/AFP/Getty Images

Updated at 22.43 GMT

6d ago 22.17 GMT 7th over: West Indies 51-2 (Charles 26, Hope 7) Rehan Ahmed’s first over disappears for 12. It started quite gently, with four singles and a
two off the first five balls, but then Charles clattered a no-nonsense straight six. “We’re not going to win this, Rob,” writes Guy Hornsby. “But I’m just here
to say I’m still watching for Woakes’ seam position and Rehan’s googly. That is all.”
</item 12>

<item 13>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.57 GMT WICKET! West Indies 113-5 (Powell c Rashid b Topley 8) England haven’t given this up. Powell edges Topley to short third man, where Rashid
takes a smart low catch to his left. West Indies need 20 from 21 balls. Reece Topley (right) celebrates the dismissal of Rovman Powell. Photograph: Randy
Brooks/AFP/Getty Images

Updated at 23.07 GMT

6d ago 22.54 GMT 16th over: West Indies 112-4 (Hope 29, Powell 8) Hope, who is playing an exemplary innings, opens the face to steer Rehan for his first
boundary. Rehan has struggled as the series has progressed, an d later in the over Powell muscles him over long-on for six. West Indies are 21 runs away from a
hat-trick of T20 series wins this year: South Africa away, India and England at home.
</item 13>

<item 14>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.05 GMT 4th over: West Indies 34-2 (Charles 15, Hope 1) England needed early wickets to have a realistic chance of victory. One more in the Powerplay,
before they bring on Adil Rashid, would put them right back in the game.



6d ago 22.03 GMT WICKET! West Indies 33-2 (Pooran b Woakes 10) UltraEdge shows Pooran did edge that ball from Topley – but it doesn’t matter because Woakes has
got him! Pooran tried to ride the bounce of an awkward delivery that cramped him for room, took the inside of an angled bat and ricocheted onto the stumps.
Nicholas Pooran looks glum as his bails go flying.
</item 14>

<item 15>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Just as surely as when they came out on Tuesday facing an almost unachievable task, the outcome of the West Indies’ run chase seemed inevitable from the start.
With such a meagre target they needed little more than to keep their heads, and though both Topley and Sam Curran conceded only two with their final overs to
push the contest into its final over, that never looked in doubt.

skip past newsletter promotion Sign up to The Spin Free weekly newsletter Subscribe to our cricket newsletter for our writers' thoughts on the biggest stories
and a review of the week’s action Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more
information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy
. We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Quick Guide How do I sign up for sport breaking news alerts?
</item 15>

<item 16>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
That’s Rashid’s 99th T20I wicket – in truth it was a bit of a drag down but from the front of the hand so it skidded onto Mayers who didn’t time it and was
caught by Brook in the deep. England needed that BIG TIME. A masterclass from Rashid in the rest of the over, Nicholas Pooran is the new man and his clip for
one is the only run from the over. 8th over: West Indies 79-2 (Hope 19, Pooran 1)

Updated at 00.56 GMT

13 Dec 2023 00.28 GMT 7th over: West Indies 78-1 (Mayers 35, Hope 19) Rehan to Kyle Mayers. First ball is dragged down and launched over the midwicket boundary
for SIX. Pressure on the young leggie. What a shot! Mayers holds the pose as he lofts a supreme drive down the ground for SIX more. Brutal elegance.
</item 16>

<item 17>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.37 GMT 9th over: West Indies 87-2 (Hope 23, Pooran 3) Will the wickets slow the Windies as they did England? Rehan is replaced by Tymal Mills who
starts with another wide. Rusty stuff so far from the southpaw seamer. That’s better – a series of slower balls and back of the hand trickery restricts Pooran
and Hope before the final ball is top edged over Buttler’s head for a one bounce four. West Indies need 85 from 66 balls.



13 Dec 2023 00.32 GMT WICKET! Kyle Mayers c Brook b Rashid 35 (West Indies 79-3) The sorcerer follows his apprentice and bags a wicket first ball!
</item 17>

<item 18>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
What a difference two days make. Somehow Tuesday’s ludicrous, record-smashing, high-scoring entertainment was followed by a sequel on the same surface that
eventually delivered thrilling low-scoring entertainment, and a different result claimed in very different style. It was eventually won for West Indies with
four balls to spare and in the most unfitting of ways – with a massive six – by Shai Hope, again proving himself the right man for a run-chase.

So a series marked by stunning power hitting, roaring run rates and an unexpected midway transformation in England’s performances and prospects concluded with a
3-2 triumph for West Indies, who extended their supremacy over England at home to four series in three formats across nearly five years.

England assembled exactly the same players in exactly the same place and exactly the same situation – losing the toss, put in to bat – but this was an entirely
different performance.
</item 18>

<item 19>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.50 GMT 15th over: West Indies 99-4 (Hope 23, Powell 1) Rashid finishes with figures of 4-0-21-2. He’s had a great series: eight wickets at 15 with an
absurd economy rate of 6.26. But he’s going to end on the losing side.



6d ago 22.48 GMT WICKET! West Indies 95-4 (Rutherford c Curran b Rashid 30) Buttler brings back Rashid for his final over, knowing it’s now or never for
England, and gets an immediate reward. Rutherford smashes a drive towards short extra, where Curran dives to his left to take a superb catch. It’s not the
despair, Laura…
</item 19>

<item 20>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
Brutal elegance. Rehan comes back well with two sharply turning leggies. Bosh! Shai Hope gets in on the act and smears a length ball over mid on for the third
six of the over. It’s a learning curve for the tyro spinner but that doesn’t make it any less bruising.



13 Dec 2023 00.23 GMT 6th over: West Indies 59-1 (Mayers 22, Hope 13) Curran is called back, Mills granted just the one over for the time being. Two slower
balls bring two soothing dots for England. A single each to Mayers and Hope. Curran does very well to get out of the over with just three runs conceded.
Powerplay complete. How long till we see some spin from Rehan? Not long at all, he’s on for the next over.
</item 20>

<item 21>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.45 GMT 14th over: West Indies 94-3 (Hope 19, Rutherford 30) West Indies are rushing towards a series victory. Rutherford hooks the new bowler Sam
Curran for six more, and there are 11 from the over in total. Rutherford has 30 from 23 balls, Hope 19 from 28. They’re playing their roles to perfection.



6d ago 22.40 GMT 13th over: West Indies 83-3 (Hope 17, Rutherford 21) Rehan Ahmed returns, perhaps in the hope of tempting Rutherford into one big shot too
many. Instead it’s Shai Hope, who is happily playing the anchor role, who faces five of the six deliveries. England are running out of time.
</item 21>

<item 22>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
That is all.”



6d ago 22.13 GMT 6th over: West Indies 39-2 (Charles 16, Hope 5) Charles mistimes a short ball from Woakes just short of Moeen in the covers. In the end Moeen
does well to save the boundary. This has been a good fightback by England, with only nine runs scored in the last three overs.



6d ago 22.09 GMT 5th over: West Indies 37-2 (Charles 15, Hope 4) Topley continues, which makes sense because the spinners will be doing most of the work after
the Powerplay, and concedes only three runs. He has quietly had a good series and is comfortably the most economical seamer on either side.
</item 22>

<item 23>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 23.09 GMT 18.4 overs: West Indies 123-5 (Hope 37, Russell 3) Sam Curran concedes only one run from the first four balls of a superb penultimate over. The
ball is getting very wet, though, so there’s a short break in play while England call for a dry towel. West Indies need 10 from 8 balls.



6d ago 23.04 GMT 18th over: West Indies 122-5 (Hope 36, Russell 3) Jos Buttler gambles by continuing with Rehan Ahmed, who has been relatively expensive. It
doesn’t come off: Hope skims a back cut between short third and backward point for four to move the requirement back below a run a ball. It’s very hard to see
how England can with this now.
</item 23>

<item 24>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Often there is a sense as a tour enters its final days of thoughts turning to home, and perhaps the focus was not as laser-sharp as it might have been for this
game, determination less than absolute to grasp this opportunity to end England’s run of reverses in the Caribbean. Whatever the reason, the pyrotechnics that
rocketed England to victory in the two previous matches never sparked in this one. It took a fabulous delivery from Gudakesh Motie to end Phil Salt’s run of
absurd form with his score a modest 38, but some of the other wickets that fell during the period that defined England’s innings and the game seemed a little
careless.

In the fourth over Jos Buttler casually turned a Jason Holder delivery to fine leg, where Oshane Thomas was positioned, hands cupped. Harry Brook tried to
paddle-sweep Motie but sent the ball only a few feet into the air, giving Nicholas Pooran an easy catch.
</item 24>

<item 25>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
The wicket has given Ahmed a confidence boost and he gets out of the over with three dots. 11th over: West Indies 101-3 (Hope 25, Hetmyer 1)



13 Dec 2023 00.43 GMT 10th over: West Indies 99-2 (Hope 25, Pooran 13) Rashid is given another after his excellent first over. The googly is spotted by Pooran
and driven down the ground handsomely for SIX! A glide behind point bring a single, Hope returns the favour with a flick into the leg side. A short ball is
panned into the deep by Pooran for a couple to end the over. Bon Jovi O’clock – West Indies need 73 from 60. Time for a quick slurp for the players and a
knuckle stretch for me. Is there anybody alive out there?!
</item 25>

<item 26>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.25 GMT ENGLAND REVIEW FOR LBW AGAINST RUTHERFORD! This looks really close. Rutherford pushed around a legbreak from Rashid and was hit on the pad in
front of middle and leg.



6d ago 22.24 GMT 9th over: West Indies 58-3 (Hope 11, Rutherford 2) Moeen replaces Rehan and rushes through a pretty good over, four singles from it. West
Indies are still on top but another wicket would make things very interesting.



6d ago 22.21 GMT WICKET! West Indies 54-3 (Charles c Buttler b Rashid 27) Adil Rashid strikes straight away. On this occasion it had nothing to do with his
bowling, but maybe a little to do with his aura. He accidentally bowled a very wide, dipping full toss to Charles, who clunked it low to Jos Buttler at short
extra cover.
</item 26>

<item 27>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
A wicket once so true turned untrustworthy and they struggled to a score of 132, less than half the 267 to which they had joyously motored just 48 hours
previously. And though the home side’s response was anything but overwhelming, neither was the result ever in significant doubt.

“We’ve had some high-scoring games, but the last game was almost the complete opposite of what happened here,” Reece Topley said. “We were probably just under
par with the first-innings score but with the ball we were excellent, pushed them right to the end. I think it’s been an amazing series. Both teams have played
some unbelievable cricket. I was so excited turning up today, because it was basically a final and those are the games you want to play in and be on the right
side of. The bottom line is you want to win this series, I want to win every series I can for England.”
</item 27>

<item 28>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Amid the wider theme of transformation Adil Rashid’s excellence remains constant. The 35-year-old, who rose to the top of the T20 bowling rankings for the first
time on Wednesday, took two wickets and conceded just 21 while Topley was even better, also claiming a pair of victims but conceding only 17. England take
plenty of positives from this series, but the result was not one.
</item 28>

<item 29>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 23.36 GMT Jos Buttler’s reaction We probably didn’t adapt quickly enough with the bat – I thought 16o w0uld have been a decent score on a used wicket. It
was a fantastic bowling effort; we did well to hang in the game as long as we did. Phil Salt has had a brilliant series. We found out lots about ourselves as a
team. [On the stop-clock experiment] It was only in the last game that we really noticed it. It’s good to keep the game moving.



6d ago 23.19 GMT England’s miserable white-ball year ends with another series defeat. Plenty of good has come from this series, though, most notably the
performance of Phil Salt. While there are a few slots to fill, most notably at No3, they are in reasonable shape ahead of nexr year’s World Cup.
</item 29>

<item 30>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
That made it 70 for four and the game was one ball into its ninth over, at which point Liam Livingstone and Moeen Ali had to focus for a while on disaster
prevention rather than entertainment.

Gudakesh Motie is embraced his his West Indies teammates after dismissing England’s in-form Phil Salt. Photograph: Randy Brooks/AFP/Getty Images

Suddenly bowlers were energised by the prospect of posting, for a change, less-than-humiliating figures. Akeal Hosein’s were outstanding, his four overs going
for just 20 and bringing the wickets of Will Jacks and Moeen, undone by a relay catch on the boundary after combining with Livingstone to add 40 for the fifth
wicket. Livingstone followed precisely two overs later, prodding the ball limply back into the hands of Motie, at which point things really fell apart, the last
five wickets surrendered in 20 balls for just 11 runs.
</item 30>
=== END ITEM id=3e4d811979722683 ===

=== ITEM id=c85cb04ee9652854 ===
Gold sentences and facts:
Sentence 0: A number of videos posted to X since the attacks have been identified as entirely unrelated to the conflict — including footage that was filmed last month in Egypt and even a clip from a video game that had been posted to the platform with a (false) claim it showed Hamas missile attacks on Israel.
  Fact 0.0: A number of videos posted to X since the attacks have been identified as entirely unrelated to the conflict.
  Fact 0.1: One unrelated video was filmed last month in Egypt.
  Fact 0.2: Another unrelated item was a clip from a video game.
  Fact 0.3: The video game clip was posted to X with a false claim that it showed Hamas missile attacks on Israel.
Sentence 1: We’ve reached out to the Commission with questions about how it plans to respond to this latest analysis of X’s role in spreading disinformation.
  Fact 1.0: The outlet reached out to the Commission with questions.
  Fact 1.1: The questions ask how the Commission plans to respond to this latest analysis of X's role in spreading disinformation.

Retrieved items:
<item 1>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
In another recent tweet-response Musk can also be seen chipping into another conversation in which an X user has commented on a screengrab of an apparent
exchange of threats between Iran’s supreme leader, Ayatollah Khamenei, and an official Israeli government account in relation to the war — with Musk writing:
“Amazing to see this exchange!”

The irony here is Musk’s destruction of Twitter’s legacy verification of notable accounts means you can’t, at a glance, be sure if the exchange really happened.
</item 1>

<item 2>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
In this case, failure to reply by the deadline could lead to the imposition of period penalties,” the EU added.

Earlier this year, Musk pulled out of the EU’s Code of Practice on online disinformation. In a response to X leaving the voluntary Code, Breton warned:
“Obligations remain. You can run but you can’t hide.”

In recent days, TikTok and Meta have also been warned by the EU over disinformation related to the Israel-Hamas war. But back in September, a study commissioned
via the EU’s Code of Practice suggested X is the worst of the major platforms when it comes to spreading disinformation.

This report was substantially revised to clarify a number of aspects, including — most saliently — that the EU has sent a formal request for information to X at
this point. This could prefigure a formal announcement of an investigation but is not, yet, that formal technical step.
</item 2>

<item 3>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
X has been pushing a so-called Community Notes feature — which crowdsources additional context to add to questionable tweets — as its main response to
disinformation risks.

By Thursday afternoon, Hamas’ attack had killed more than 1,200 people, per the Israeli Military, and at least 1,537 people had been killed in Gaza by Israel’s
retaliatory strikes, according to the Gaza Ministry of Health.

Returning to the EU process, X has until October 18 to provide the bloc with information about “the activation and functioning of X’s crisis response protocol”;
and until October 31 to respond to its other requests.

“Based on the assessment of X replies, the Commission will assess next steps,” the Commission wrote. “This could entail the formal opening of proceedings
pursuant to Article 66 of the DSA.”

It also noted the regulation includes powers to impose fines for “incorrect, incomplete or misleading information in response to a request for information”.

“In case of failure to reply by X, the Commission may decide to request the information by decision.
</item 3>

<item 4>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
the so-called Blue checks”

“If proven, these failures would constitute infringements of Articles 34(1), 34(2) and 35(1), 16(5) and 16(6), 25(1), 39 and 40(12) of the DSA,” the EU added —
saying it will now carry out an in-depth investigation “as a matter of priority”.

Today we open formal infringement proceedings against @X : ⚠️ Suspected breach of obligations to counter #IllegalContent and #Disinformation ⚠️ Suspected breach
of #Transparency obligations ⚠️ Suspected #DeceptiveDesign of user interface#DSA pic.twitter.com/NxKIif603k — Thierry Breton (@ThierryBreton) December 18, 2023

The Commission is responsible for enforcing the DSA on larger platforms such as X — which was designated a very large online platform (VLOP) under the DSA back
in April.
</item 4>

<item 5>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Under Musk, X has pushed the use of an existing crowdsourced fact-checking feature (rebranded as Community Notes) which essentially seeks to outsource the issue
to a hands-length process of aggregating views from platform users, some of which may be appended as contextual notes to dubious tweets. At the same time the
erratic billionaire has set about removing signals Twitter users were previously able to rely on to help assess the quality of information tweets contained,
such as removing state-affiliated media labels from propaganda outlets and replacing legacy account verifications with pay-to-play.

X was contacted for a response to the EU’s analysis of the ratio of disinformation it’s spreading. It replied with an automated email reading: “Busy now, please
check back later.”
</item 5>

<item 6>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
We also clarified the scale of financial penalties possible under the DSA if a breach is confirmed; and added details of the regulation’s crisis response
mechanism. We also fixed an error in the original report which referred to Interpol, rather than Europol.
</item 6>

<item 7>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
They must also clearly communicate their T&Cs to users and be able to demonstrate proper enforcement.

Larger platforms like X also have additional obligations to identify and mitigate systemic risks such as disinformation, gender-based violence or negative
effects on the exercise of fundamental rights.

Additionally, the regulation includes a “crisis response” mechanism (Article 36) which enables the Commission to adopt rapid-fire measures on larger platforms
in situations of “serious threat”, such as war.

Since Saturday’s attacks in Israel, posts identified as false have been spotted circulating on X — including, in one example, a clip that purported to show
Hamas missile attacks on Israel but was actually footage from a video game.

Meanwhile X’s ability to respond internally to reports of content problems have been drastically pared back after Musk’s takeover last year — which saw major
layoffs, including in content moderation and human rights as part of the new billionaire owner’s bid to improve the platform’s profitability.
</item 7>

<item 8>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
It replied with an automated email reading: “Busy now, please check back later.”

Update: X’s global government affairs account has posted a thread responding to the EU’s criticism in which it claims the data shows other social networking
services saw greater changes in subscriber growth and total growth for pro-Kremlin accounts, and for engagement and volume; it also writes that: “More than 700
unique Community Notes have appeared on posts related to the Ukraine conflict. Through Community Notes’ new ‘Notes on Media’ feature, which automatically
matches notes on an image or video to other posts that contain matching media, these notes appeared on over 1,200 additional posts. These notes are in numerous
languages and are written specifically for local audiences.”

Back in April, X/Twitter was named as one of 19 larger platforms that face the strictest level of regulation under the EU’s DSA.
</item 8>

<item 9>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Time will tell how this one plays out but he may find the dead blue bird he’s chosen to hang around his neck, when he purposefully ripped up the standard
content moderation rulebook and opted to channel online chaos, starting to feel like it’s dragging him down.
</item 9>

<item 10>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
In a very swift test of the European Union’s newly updated content moderation rulebook, the bloc has fired a public warning at Elon Musk–owned X (formerly
Twitter) for failing to tackle illegal content circulating on the platform in the wake of Saturday’s deadly attacks on Israel by Hamas terrorists based in the
Gaza Strip.

The European Commission has also raised concerns about the spread of disinformation on X related to the terrorist attacks and their aftermath.

Unlike terrorism content, disinformation is not illegal in the EU per se. However, the EU’s Digital Services Act (DSA) puts an obligation on X — as a so-called
very large online platform — to mitigate risks attached to harmful falsehoods as well as to act diligently on reports of illegal content.

Following the terrorist attacks by Hamas against 🇮🇱, we have indications of X/Twitter being used to disseminate illegal content & disinformation in the EU.
</item 10>

<item 11>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
EU regulators are also set to assess Musk & co’s claims that replacing a formal content moderation function with crowdsourced opinions is a valid approach to
tackling diverse information risks; and, consequently, to look into how much Musk’s gutting of content moderation teams and processes has cost X’s users in
terms of harm exposure.

To say the Commission’s investigation looks fascinating and timely is an understatement.

EU lawmakers can certainly bask in the glow of being prepared for the possibility of an erratic billionaire running a wrecking ball through a major social
platform. But of course they will need to follow through — and it’s clear the real work is just starting. So this probe looks set to be a test of how much
stomach the EU has to actually enforce the rules it loves to lay down on paper.

The investigation may also test Musk’s mettle for what could be an expensive head-on clash with EU regulators.
</item 11>

<item 12>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
Per Mekić, the Commission’s ad campaign, which apparently racked up millions of views on X, insinuated that opponents of the proposed legislation did not want
to protect children — messaging he dubbed “a form of emotional blackmail”.

The ads included what he suggested is a misleading claim that the majority of Europeans support the proposal — but which is based on a survey that highlighted
“only the benefits but not the drawbacks of the proposed legislation”. Other surveys, by research firms YouGov and Novus, that highlighted the drawbacks showed
“virtually no support” for the plan among the European population, his post also pointed out.
</item 12>

<item 13>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Reached for a response to the EU probe, X’s Joe Benarroch — who was brought into the company by CEO Linda Yaccarino relatively recently to work on business
operations — said:

X remains committed to complying with the Digital Services Act, and is cooperating with the regulatory process. It is important that this process remains free
of political influence and follows the law. X is focused on creating a safe and inclusive environment for all users on our platform, while protecting freedom of
expression, and we will continue to work tirelessly towards this goal.

While X under Musk has very evidently been flying the ship in the opposite direction to the responsible governance the DSA intends to encourage — hence
attracting so many warnings from EU lawmakers, even long before today’s formal probe — it’s notable X recently started offering a research program for outsiders
to get data to systemic risks arising on the platform in the EU. However the Commission obviously has doubts X has gone far enough on the transparency front to
meet the DSA’s bar.
</item 13>

<item 14>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Confirmed breaches of the online governance regime can face a range of major sanctions, including fines of up to 6% of global annual turnover.

The EU can also apply interim measures where it believes there’s a risk of serious harm for users and may even seek to have access to infringing services
temporarily blocked. Although its announcement today is careful to caveat the development by affirming no conclusions have yet been reached.

There is also no confirmed timeline for the probe to conclude. But the fact of it being opened means other tools are put within reach of regulators and could be
applied if the EU sees the need to act quickly — such as the aforementioned interim measures. So the Commission’s official scrutiny of X could have real world
implications for how the platform operates sooner rather than later.
</item 14>

<item 15>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
Asking Musk to be “effective” against disinformation is a bit like asking the sea to stop moving. But, well, this is how the regulatory dance must go (and after
the dance comes the denouement — which, if a DSA breach is confirmed, means enforcement. And actual penalties might be a bit harder for Musk to troll).

In the meantime, the EU has asked Musk to make contact with relevant law enforcement authorities and Europol — and “ensure that you respond promptly to their
requests.” Breton also flags some unspecified additional DSA compliance issues he says his team will be contacting Musk’s about “shortly” — “with a specific
request.”. (We’ve asked the EU what else it’s concerned about on X and will update this report with any response.)

The bloc has given Musk a deadline of 24 hours to respond to its asks at this point — stipulating his answer will be added to its assessment file on X’s
compliance with the DSA.
</item 15>

<item 16>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
A number of videos posted to X since the attacks have been identified as entirely unrelated to the conflict — including footage that was filmed last month in
Egypt and even a clip from a video game that had been posted to the platform with a (false) claim it showed Hamas missile attacks on Israel.

A Wired report yesterday summed up the chaotic situation playing out on Musk’s platform in an article entitled “The Israel-Hamas War Is Drowning X in
Disinformation.”

At one point, Musk himself even recommended people follow accounts that had posted antisemitic comments and false information in the past — although he
subsequently deleted the tweet where he had made the suggestion.
</item 16>

<item 17>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
Going into more detail of the microtargeting used by the Commission, Mekić wrote: “X’s Transparency Report shows that the European Commission also used
‘microtargeting’ to ensure that the ads did not appear to people who care about privacy (people interested in Julian Assange) and eurosceptics (people
interested in ‘nexit’, ‘brexit’ and ‘spanexit’ or in Victor Orbán, Nigel Farage, or the German political party AfD). For unclear reasons, people interested in
Christianity were also excluded.

“After excluding critical political and religious groups, X’s algorithm was set to find people in the remaining population who were indeed interested in the ad
message, resulting in an uncritical echo chamber. This microtargeting on political and religious beliefs violates X’s advertising policy, the Digital Services
Act – which the Commission itself has to oversee — and the General Data Protection Regulation [GDPR].”
</item 17>

<item 18>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
A rule of law clash between Musk and the EU has looked increasingly inevitable in recent months — and, indeed, has been predicted by some industry watchers
since rumors of the erratic billionaire’s plan to take over Twitter emerged last year.

Reminder: Penalties for confirmed breaches of the DSA can be as high as 6% of global annual turnover. The bloc also has powers, in extremis, which could result
in access to X being shut off in the region if the platform repeatedly fails to correct course. So if Musk fails to satisfy EU regulators, he could face serious
consequences for what remains a highly indebted business.

In his letter to Musk, Breton writes that the EU has “indications” that X is being used to disseminate illegal content and disinformation in the EU” following
Saturday’s attacks. He then goes on to remind the company of the DSA’s “very precise obligations” vis-à-vis content moderation.
</item 18>

<item 19>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
The EU said the probe will also look at the effectiveness of “related policies mitigating risks to civic discourse and electoral processes”

— with a focus on the effectiveness of its “Community Notes” system of crowdsourcing fact-checking to aggregated views of users, who may vote to label disputed
tweets with additional context. The EU said the probe will also look at the Transparency requirements — with a focus on “suspected shortcomings” by X in giving
researchers access to publicly accessible data as mandated by Article 40 of the DSA, and suspected shortcomings in its ads repository (aka the ads transparency
library the regulation also mandates)

— with a focus on “suspected shortcomings” by X in giving researchers access to publicly accessible data as mandated by Article 40 of the DSA, and suspected
shortcomings in its ads repository (aka the ads transparency library the regulation also mandates) Suspected deceptive design elements in X’s user interface —
“notably in relation to checkmarks linked to certain subscription products, the so-called Blue checks”

“If proven,
</item 19>

<item 20>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
During this period the DSA had not entered into force yet for VLOPs and VLOSEs, including X/Twitter. The Commission cannot comment further on the substance of
the risk assessments. We are analysing the available information on a range of issues and we are in close contact with X but we cannot comment on specifics and
timelines. As stated by Commissioner [Thierry] Breton, minor protection and integrity of elections are priorities in the enforcement of the DSA.

Penalties for breaches of the DSA can scale up to 6% of global annual turnover and the Commission has previously wanted that not following the Code of Practice
on Disinformation would be factored into its assessments. So the price for X ignoring the EU’s warnings on disinformation could be steep — albeit its ad revenue
has cratered since Musk’s takeover, as a result of his failure to tackle toxicity and harmful disinformation encouraging an advertiser exodus — and shrinking
earnings would effectively downsize any future penalties X could face under the DSA.
</item 20>

<item 21>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
X should have “proportionate and effective mitigation measures” in place to tackle “the risks to public security and civic discourse stemming from
disinformation,” he also says.

Instead, the platform appears to be turning into an engine of disinformation — one that’s demonstrably enabled the very swift amplification of a smorgasbord of
toxic fakes around the Israel-Hamas war. Falsehoods that may be trying to manipulate opinion around the conflict, or otherwise exploit horrific events to drive
engagement (clickbait) or for even more bleak, cynical and potentially harmful ends.

“Public media and civil society organisations widely report instances of fake and manipulated images and facts circulating on your platform in the EU, such as
repurposed old images of unrelated armed conflicts or military footage that actually originated from video games. This appears to be manifestly false or
misleading information,” writes Breton. “I therefore invite you to urgently ensure that your systems are effective, and report on the crisis measures taken to
my team.”
</item 21>

<item 22>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Are the Texas and Florida laws identical?

The two laws — HB 20 in Texas and SB 7072 in Florida — are very similar in their origins and intentions, but diverge slightly in how they seek to restrict
social media platforms.

In both instances, a provision of the state law instructed social media companies to stop removing certain kinds of content. In Texas, the law told social media
companies that they could no longer remove or demonetize content based on the “viewpoint represented in the user’s expression.” In Florida, the law would stop
social media companies from banning political candidates or removing or restricting their content. The laws have a few other provisions, but the idea is that
conservative politicians in those states want to regulate how tech companies interact with political content.

“They’re in the same ballpark, the sentiment is the same,” Barrett said.
</item 22>

<item 23>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
It also puts legal obligation on larger platforms — including X — to mitigate risks from disinformation. So the fast-moving and bloody events unfolding in
Israel and Gaza are offering a real-world test of whether the EU’s rebooted rulebook is big and beefy enough to tackle X’s most notorious shitposter. Who, since
last fall, is also the platform’s owner.

Since taking over Twitter (as it was then), Musk has painted the largest target on X when it comes to DSA enforcement on account of a series of changes he’s
pushed out that make it harder for users to locate quality information on X.

This includes ending legacy account verification and turning the Blue Check system into a game of pay-to-play.
</item 23>

<item 24>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Per the Commission, the EU’s investigation of X focuses on the following areas and issues:

Compliance with DSA obligations related to countering the dissemination of illegal content in the EU — “ notably in relation to the risk assessment and
mitigation measures adopted by X to counter the dissemination of illegal content in the EU, as well as the functioning of the notice and action mechanism for
illegal content in the EU mandated by the DSA, including in light of X’s content moderation resources

— “ The effectiveness of measures taken by X to combat info manipulation — with a focus on the effectiveness of its “Community Notes” system of crowdsourcing
fact-checking to aggregated views of users, who may vote to label disputed tweets with additional context.
</item 24>

<item 25>
Meta says it’s prioritizing livestreaming checks during Israel-Hamas war (TechCrunch, 2023-10-13T12:43:45+00:00)
Penalties for failing to comply with the pan-EU regulation can reach as high as 6% of global annual turnover — which, in Meta’s case, could run to multiple
billions.

The social media giant is not alone in being warned by the bloc over content concerns attached to the Israel-Hamas war: Elon Musk’s X has been singled out for
even greater attention here — with the bloc issuing both an “urgent” warning earlier this week and following that with a formal request for information about
its compliance approach.

TikTok has also received a warning from the EU about DSA content risks related to the conflict.

This report was updated with details about the operational timeline for the DSA crisis response mechanism coming into operation
</item 25>

<item 26>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
But at the time of writing, Musk had engaged in a bit of a dance of his own with the news of the EU’s warning by responding to a critical tweet posted to X by
journalist Glenn Greenwald, who attacked the EU’s new “censorship law,” as he dubbed the DSA — which he claimed would be used to “punish X.”

In reply to Greenwald, Musk avoided expressing the same trenchant criticism of the EU’s approach but invited yet more relativism — writing: “Let the public hear
exactly what this disinformation consists of and decide for themselves.”

He then went on to sew doubt that disinformation is something that can be independently arbitrated at all, implying fact-checking is just a convenient exercise
for targeting different opinions — so essentially echoing Greenwald’s position — by claiming that “many times” the “official fact-checker” has been found making
false statements, before adding a rhetorically empty offer that “Maybe this is the case here, maybe not” for good, plausible deniability measure.
</item 26>

<item 27>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
The law puts an obligation on so-called very large online platforms (VLOPs) to assess societal risks attached to their use of algorithms — and put in place
“reasonable, proportionate and effective mitigation measures” for identified risks, with their reporting and mitigation plans subject to independent audit and
oversight by the Commission.

We’ve reached out to the Commission with questions about how it plans to respond to this latest analysis of X’s role in spreading disinformation.

Update: A Commission spokesperson told us:

While the Code is voluntary, compliance with the DSA is not. The DSA is now legally binding on X/Twitter. This includes in particular the obligation of Very
Large Online Platforms, such as X/Twitter to mitigate the risks their platforms pose for the spread of disinformation. The TrustLab study to measure two
Structural Indicators covers signatories of the Code at the time of the launch of the exercise, which at the time included X/Twitter.
</item 27>

<item 28>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
X (formerly Twitter) has been called out in the European Union for having the worst ratio of disinformation/misinformation to posts not spreading falsehoods
among mainstream social networks.

In a speech today discussing the latest updates from other platforms that have signed up to combat disinformation under a beefed up EU Code of Practice that’s
been linked to compliance with the the bloc’s legally binding Digital Services Act (DSA), it commissioner for values and transparency, Vera Jourova, said X’s
poor performance was assessed during a pilot phase of testing of new methodology developed by Code signatories.

“A big novelty under the Code is now the publication of the first set of structural indicators such as how easy it is to find disinformation content, how much
engagement such content receives or indicators about sources. These indicators — developed by the signatories — constitute an unprecedented and novel insight
into disinformation on online platforms,” she suggested. “Such insight is crucial to understand how effective platforms efforts are to fight this threat even
more efficiently.
</item 28>

<item 29>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
The EU also said then it was looking at X’s compliance across a number of areas, including policies and practices regarding notices on illegal content,
complaint handling, risk assessment and measures to mitigate the risks identified — and it gave X until the end of October to respond.

“On the basis of the preliminary investigation conducted so far, including on the basis of an analysis of the risk assessment report submitted by X in
September, X’s Transparency report published on 3 November, and X’s replies to a formal request for information, which, among others, concerned the
dissemination of illegal content in the context of Hamas’ terrorist attacks against Israel, the Commission has decided to open formal infringement proceedings
against X under the Digital Services Act,” the EU said today.
</item 29>

<item 30>
The UK just laid out new rules for the internet — it only gets harder from here (The Verge, 2023-11-09T00:01:00+00:00)
Ofcom recommends these platforms implement policies like not allowing strangers to send direct messages to children, using hash matching to detect and remove
CSAM, maintaining content and search moderation teams, and offering ways for users to report harmful content.

Large tech platforms already follow many of these practices, but Ofcom hopes to see them implemented more consistently. “We think they represent best practice
of what’s out there, but it’s not necessarily applied across the board,” Whitehead says. “Some firms are applying it sporadically but not necessarily
systematically, and so we think there is a great benefit for a more wholesale, widespread adoption.”

There’s also one big outlier: the platform known as X (formerly Twitter). The UK’s efforts with the legislation long predate Elon Musk’s acquisition of Twitter,
but it was passed as he fired large swaths of its trust and safety teams and presided over a loosening of moderation standards, which could put X at odds with
regulators.
</item 30>

<item 31>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
“When you receive notices of illegal content in the EU, you must be timely, diligent and objective in taking action and removing the relevant content when
warranted,” he warns. “We have, from qualified sources, reports about potentially illegal content circulating on your service despite flags from relevant
authorities.”

He also takes issue with a change X made last night to its public interest policy, under which it judges newsworthiness (i.e., for deciding whether posts that
infringe its rules may nonetheless remain on the site) — but which Breton says have left “many European users uncertain” (i.e., about how X is applying its own
rules).

Again, this is a problem because the DSA requires platforms to be clear and transparent about their rules and how they apply them. “This is particularly
relevant when it comes to violent and terrorist content that appears to circulate on your platform,” Breton continues in another pointed warning.
</item 31>

<item 32>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“Upcoming national elections and the EU elections will be an important test for the Code that platforms signatories should not fail,” said Jourova today,
warning: “Platforms will need to take their responsibility seriously, in particular in view of the DSA that requires them to mitigate the risks they pose for
elections.

“The DSA is now binding, and all the VLOPs have to comply with it. The Code underpins the DSA, because our intention is to transform the Code of Practice into a
Code of Conduct that can form part of a co-regulatory framework for addressing risks of disinformation.”

A second batch of reports by disinformation Code signatories have been published today, covering the January to June period. At the time of writing only a
handful are available for download on the EU’s Disinformation Code Transparency Center — including reports from Google, Meta, Microsoft and TikTok.

The EU said these are the most extensive reports produced by signatories to the Code since it was set up back in 2018.
</item 32>

<item 33>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
And also today, as we reported earlier, Jourova drew attention to early analysis conducted by some of the remaining signatories which she said had found X
performed the worst for disinformation ratios.

This suggests that X, which back in April was designated by the EU as a VLOP under the DSA, continues to put itself squarely in the Commission’s crosshairs —
including over its priority issue of tackling Kremlin propaganda.

As well as devising the anti-disinformation Code, the bloc’s executive is now responsible for oversight of VLOPs’ compliance with the DSA — with powers under
the new law to fine violators up to 6% of global annual turnover.
</item 33>

<item 34>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
He’s also ripped up a bunch of legacy content moderation policies and slashed in-house enforcement teams while promoting a decentralized, crowdsourced
alternative (rebranded as Community Notes), which essentially outsources responsibility for dealing with tricky issues like disinformation to users in what
looks suspiciously like another gambit to eke out extra engagement and farm confusion by applying a philosophy of extreme relativism so culture warriors are
encouraged to keep forever fighting for their own “truth” in the comments.

Oh and he also pulled X out of the EU’s Code of Practice on Disinformation earlier this year in a very clear thumb of the nose to EU regulators.

Urgent letter to Musk

In an “urgent” letter to Musk today, which the EU’s internal market commissioner Thierry Breton also shared on X, the bloc has sent the strongest signal yet it
believes Musk’s platform is in breach of the DSA — although this is by no means Musk’s first warning.
</item 34>

<item 35>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
The European Union’s home affairs commissioner, Ylva Johansson, has confirmed the Commission is investigating whether or not it broke recently updated digital
governance rules when her department ran a microtargeted political ad campaign aiming to drive support for a controversial child sexual abuse material
(CSAM)-scanning proposal she’s spearheading.

But at a committee hearing in the European Parliament today she deflected MEPs’ enquiries for her to give more details about the ad campaign.

The governance regulation concerned is the Digital Services Act (DSA), which includes provisions relating to online advertising — including a prohibition on the
use of sensitive personal data, such as political opinions, for targeting ads. While the ads in question ran on X (formerly Twitter) — which is already expected
to be compliant with the DSA, having been designated by the Commission as a so-called Very Large Online Platform (VLOP) back in April.

The Commission itself, meanwhile, not only proposed this pan-EU law but is responsible for oversight of VLOPs’ DSA compliance.
</item 35>

<item 36>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
This is a very valuable industry proposal that has great potential while it has to be still further developed.

“Here, signatories have decided to run a pilot phase in three Member States to evaluate the methodology on two of those indicators. X, former Twitter, who is
not under the Code any more is the platform with the largest ratio of mis/disinformation posts.”

The Elon Musk-owned platform exited the EU’s Code of Practice on Disinformation back in May, shortly after EU lawmakers warned that policy shifts executed after
Musk’s takeover were boosting Kremlin propaganda and criticised the company for patchy reporting. The EU’s top diplomat also took a public shot at Twitter for
ending free access to its APIs for researchers, saying the action could threaten the study of disinformation.

Last year the Commission also warned Musk over “huge work” ahead if the platform was to avoid falling foul of the incoming DSA.
</item 36>

<item 37>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
In response to the EU’s warning earlier this week, X CEO Linda Yaccarino released a letter saying that a leadership group had been convened to consider X’s
response and “tens of thousands” of pieces of content had been removed, along with “thousands” of posts and “hundreds” of accounts linked to terrorist groups,
violence or extremism.

She also said the company formerly known as Twitter is responding to law enforcement requests but claimed it had not received any requests from Europol at the
time.

Today @lindayaX responded to @ThierryBreton’s letter detailing our work in response to the terrorist attack on Israel. pic.twitter.com/yZtaOVGpHG — Global
Government Affairs (@GlobalAffairs) October 12, 2023

In further remarks via X’s safety account, the company reported that there had been more than 50 million posts globally over the past two days that referenced
the weekend’s terrorist attack, underlining the scope of content generated.
</item 37>

<item 38>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
So — tl;dr — if EU officials have used X’s ad-targeting tools to break the bloc’s own digital rulebook it’s the very definition of an awkward situation.

The existence of the Commission’s microtargeted ad campaign seeking to drum up support for its proposed CSAM-scanning law was spotted last month by
technologist, Danny Mekić. An article with his findings ran in Dutch newspaper, De Volkskrant, earlier this month.

Using public ad transparency tools the DSA requires VLOPs to provide, Mekić found the Commission had run a paid advertising campaign on X, targeting users in
the Netherlands, Sweden, Belgium, Finland, Slovenia, Portugal and the Czech Republic — countries that were not supportive of Johansson’s CSAM-scanning proposal
according to leaked minutes from a September 14 meeting of the European Council, a co-legislative body that’s involved (along with MEPs) in determining the
final shape of the CSAM law.
</item 38>

<item 39>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“Today, this is a multi-million euro weapon of mass manipulation aimed both internally at the Russians as well as at Europeans and the rest of the world. We
must address this risk. The very large platforms must address this risk. Especially that we have to expect that the Kremlin and others will be active before
elections. I expect signatories to adjust their actions to reflect that there is a war in the information space waged against us and that there are upcoming
elections where malicious actors will try to use the design features of the platforms to manipulate.”

Per the Commission’s early analysis of Big Tech’s Code reports, YouTube shut down more than 400 channels between January and April 2023 which were involved in
coordinated influence operations linked to the Russian-state sponsored Internet Research Agency (IRA). It also removed ads from almost 300 sites linked to
state-funded propaganda sites.

While the EU highlighted that TikTok’s fact-checking efforts now cover Russian, Ukrainian, Belarusian and 17 European languages, including through a new
partnership with Reuters.
</item 39>

<item 40>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
EU turns up the heat on X over illegal content in wake of Israel-Hamas war

The European Union is cranking up the heat on Elon Musk-owned X. Late Thursday, the Commission sent the company a formal request for more information — after
issuing a public warning Wednesday about reports that illegal content and disinformation targeting the Israel-Hamas war is circulating on the platform, and
Twitter’s CEO sending a high-level (non specific) response to that warning.

The move could prefigure the opening of a formal investigation of X under the bloc’s content moderation rulebook, the Digital Services Act (DSA). If the
Commission moves ahead on that, it will be the first investigation opened under the DSA since a compliance deadline for so called “very large online platforms”
(aka VLOPs) kicked in this summer.

Non-compliance under DSA rules, meanwhile, could trigger fines of up to 6% of annual turnover, plus potential blocking of a service for repeated infringements.
X owner Elon Musk last projected revenues of about $3 billion for this year.
</item 40>

<item 41>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
X owner Elon Musk last projected revenues of about $3 billion for this year. It’s anyone’s guess if that’s accurate now, but as an example, that would result in
a fine of up to $180 million.

Indirectly, an investigation will also be a key way for the public to have significantly more transparency around how X is being used (and abused): since Musk
took Twitter private, the company no longer produces quarterly earnings reports and is not held to account as closely as a result.

EU officials confirmed to TechCrunch that an investigation has not been taken at this point. Clearly, though, yesterday’s development (and the pace at which
regulators are moving) is a strong indication of the bloc’s direction of travel.

The Commission said it is looking at X’s compliance with the DSA across a number of areas — including with regard to its policies and practices regarding
notices on illegal content; complaint handling; risk assessment and measures to mitigate the risks identified.
</item 41>

<item 42>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
“I remind you that following the opening of a potential investigation and a finding of non-compliance, penalties can be imposed,” Breton adds, concluding the
letter.

We’ve asked the Commission to confirm whether it has opened an investigation into X’s DSA compliance over concerns it’s raised in the letter. Perhaps it’s going
to wait a day to see what his response is before taking that next dance step.

Musk’s extremely iterative — or just plain erratic/arbitrary — management style is so far from the responsive and responsible qualities the EU’s rulebook
demands of digital leaders it’s hard to see how this clash can end well for either side.

We contacted X for a response to the EU’s warnings about its DSA compliance but at press time the company had not responded — beyond firing back its usual
automated reply, which reads: “Busy now, please check back later.”
</item 42>

<item 43>
The UK just laid out new rules for the internet — it only gets harder from here (The Verge, 2023-11-09T00:01:00+00:00)
Ofcom’s guidelines, for example, specify that users should be able to easily block users — but Musk has publicly stated his intentions to remove X’s block
feature. He’s clashed with the EU over similar rules and reportedly even considered pulling out of the European market to avoid them. Whitehead declined to
comment when I asked whether X had been cooperative in talks with Ofcom but said the regulator had been “broadly encouraged” by the response from tech firms
generally.

“We think they represent best practice of what’s out there, but it’s not necessarily applied across the board.”

Ofcom’s regulations also cover how sites should deal with other illegal harms like content that encourages or assists suicide or serious self-harm, harassment,
revenge porn and other sexual exploitation, and the supply of drugs and firearms. Search services should provide “crisis prevention information” when users
enter suicide-related queries, for example, and when companies update their recommendation algorithms, they should conduct risk assessments to check that
they’re not going to amplify illegal content.
</item 43>

<item 44>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
In a blog post ahead of today’s hearing — which deploys a crisis-PR-esque headline claim of “setting the record straight” — she criticized the article as “a
series of insinuations looking for a home”; claiming it paired an outline of “a selection of meetings I had, of events I attended, or conferences I addressed”
with “a conspiratorial tone” in an attempt “to create the impression of financial influence where there is none”.

Fotiadis was asked by the LIBE committee about the accusation that the journalists had, essentially, been spreading disinformation — and specifically whether he
believed Johansson and the Commission’s response to it amounted to a restriction on media freedom. He responded by saying he did not think that was the case.
But went on to express surprise at how the Commission had reacted to the scrutiny — to its instinct to deploy “spin-doctor” tactics to try to discredit the
article, rather than engaging with the substance of the concerns being raised.
</item 44>

<item 45>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“In this context, 832 videos related to the war have been fact-checked, of which 211 have been removed,” Jourova noted.

The EU also flagged reporting by Microsoft that told it Bing Search had either promoted information or downgraded questionable information in relation to almost
800,000 search queries related to the Ukraine crisis.

Jourova’s speech also highlighted a couple of other areas where she urged Code signatories to go further — calling (yet again) for more consistent moderation
and investment in fact-checking, especially in smaller Member States and languages.

She also criticized platforms over access to data, saying they must step up efforts to make sure researchers are empowered to scrutinize disinformation flows
“and contribute to the necessary transparency”.

Both are areas where X/Twitter under new owner, Elon Musk, has moved out of step with EU expectations on countering disinformation.

Twitter (now X) was an original signatory to the disinformation Code but Musk took the platform out of the initiative back in May, as critical scrutiny of his
actions dialled up in the EU.
</item 45>

<item 46>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
This video certainly doesn't show a new air assault on Israel by Hamas militants, because it's actually from the video game Arma 3. pic.twitter.com/IcX7xs8QDp —
Shayan Sardarizadeh (@Shayan86) October 9, 2023

Last night, we saw a totally false claim about Israel bombing a church in Gaza go viral, thanks to multiple blue tick accounts repeating an unverified claim
that had no evidence to back it up. Musk has created a fundamental issue with Twitter's credibility in moments of crisis. https://t.co/T70G2oNDZH — Eliot
Higgins (@EliotHiggins) October 10, 2023

The problem for Musk is the DSA regulates how social media platforms and other services that carry user-generated content must respond to reports of illegal
content like terrorism.

It also puts legal obligation on larger platforms — including X — to mitigate risks from disinformation.
</item 46>

<item 47>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
It’s official! Elon Musk’s X marks the spot of the first confirmed investigation opened by the European Union under its rebooted digital rulebook, the Digital
Services Act (DSA).

Announcing the opening of a “formal proceeding” today, the European Commission said the investigation will look at whether the social networking platform may
have breached rules linked to risk management, content moderation, dark patterns, advertising transparency and data access for researchers.

The opening of a formal DSA investigation on X comes hard on the heels of a complaint against X’s adtech by privacy rights group, noyb — although today’s formal
proceeding is unlikely to be directly linked as the Commission has been probing the platform for months via a flurry of requests for information. Its earlier
actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas war.

Back in October the Commission sent Musk’s company an “urgent” formal request for information about how it was responding to information risks arising out of
the conflict.
</item 47>

<item 48>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Subscription revenues to a revamped Twitter Blue paid offering (now called X Premium) don’t appear to have been substantial enough to plug the ad gap — which
may explain Musk’s recent suggestion he plans to charge all users a small fee. However ending free access to X could be the final nail in the coffin of the
platform as a real-time information network if it triggers a further/final exodus of remaining legacy users whose knowledge and engagement lent Twitter its
original value.

Disinformation and spammers may rush in to fill the gap, generating a parody of activity, but without a critical mass of genuine users to pay attention and
contribute value and vitality — aka the life-blood of a bona fide information network — X really will mark the spot of Twitter’s demise.

Also today, a report by misinformation watcher NewsGuard found that a decision by Musk to remove labels denoting state-run or government-affiliated media
accounts led to a surge in propaganda spreading on X, with Russia’s RT seeing the biggest gains per the analysis.

From the report:
</item 48>

<item 49>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
On Wednesday, in an “urgent” letter to Musk, the EU’s internal market commissioner Thierry Breton said the bloc had seen “indications” from “qualified sources”
that X is being used to disseminate illegal content and disinformation in the EU” following Saturday’s attacks — before reminding X of the DSA’s “very precise
obligations” vis-à-vis content moderation.

“When you receive notices of illegal content in the EU, you must be timely, diligent and objective in taking action and removing the relevant content when
warranted,” Breton also warned then.

In its latest press release, the EU said it has sent X a formal request for information under the DSA.

“This request follows indications received by the Commission services of the alleged spreading of illegal content and disinformation, in particular the
spreading of terrorist and violent content and hate speech,” it wrote, adding that the ask also addresses other aspects of DSA compliance.

The pan-EU regulation puts a series of governance obligations on digital services and platforms with the aim of ensuring companies are responding to reports of
illegal content.
</item 49>

<item 50>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
In the event MEPs got some bare bones detail.

Asked for a list of these contacts, the commissioner responded that she’d met with Google six times; Microsoft, Meta and TikTok three times each; twice with
Twitter (X); and once apiece with Apple and Amazon. She also said she’d met with the child safety organizations Thorn (twice) and Brave Movement (twice); and
with Tech Alliance and ICANN once apiece.

In wider responses related to concerns about how much commercial interest had influenced the Commission, Johansson highlighted her decision for the
CSAM-scanning proposal to be “technology neutral” — meaning the draft regulation does not support any specific tech solution — with the suggestion being EU
lawmakers had resisted lobbying by companies for a law that would explicitly favor their existing tech tools.

She also denied that only Thorn and Microsoft have technology “that is necessary for the scanning” — claiming that’s “absolutely not true”.

“There are no specific technologies mentioned [in the proposal] and I think this is an important part.
</item 50>
=== END ITEM id=c85cb04ee9652854 ===
