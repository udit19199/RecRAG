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

This file has 3 items.

=== ITEM id=e13478200002b51f ===
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
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 01.51 GMT 15th over: West Indies 125-6 (Powell 12, Russell 2) Andre Russell the man to face the hattrick ball… Boo! A decent ball on the stumps is
patted back by the big man. Drama in Barbados, Russell works for a couple. Ahmed takes his cap with a broad smile on his face – he hasn’t had it all his own way
but finishes with 3-39 from his set of four. Advantage England? 47 needed from 30 balls.

Updated at 01.54 GMT

13 Dec 2023 01.47 GMT WICKET! Shepherd c Curran b Rehan Ahmed 0 (West Indies 123-6) Rehan Ahmed on a hattrick! Buttler brings Curran into slip and Shepherd
pokes a leggie to him off his very first ball. Great stuff from Rehan who has turned the game England’s way in the space of a minute.
</item 1>

<item 2>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 01.14 GMT Promising news – the rain seems to be tailing off and a rope is being dragged around the outfield to take off any excess water. I’ll bring
news as soon as it wafts over from Barbados to my sofa.



13 Dec 2023 01.00 GMT Rain stops play (West Indies 108-4 and require 64 runs off 44 balls) What’s all this then? Out of nowhere the heavens crack like an inky
brûlée and rain starts to pour. The players scurry off as the covers are hauled on. Crucially – West Indies are six runs ahead of the Duckworth Lewis Stern
rate. IF the rain doesn’t abate – and it looks heavy at the moment – then the home side will take the spoils.
</item 2>

<item 3>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Nicholas Pooran looks glum as his bails go flying. Photograph: Ricardo Mazalán/AP

Updated at 22.15 GMT

6d ago 21.59 GMT 3rd over: West Indies 30-1 (Charles 13, Pooran 9) Nicholas Pooran strolls to the crease and launches his second ball over midwicket for six.
And why not. England thought he had edged the next ball to the keeper, though eventually they ran out of time to review. West Indies need 103 from 17 overs.
“Evening Rob,” says Kim Thonger. “Jolly exciting this game, in a damaged sort of way. It occurs to me perhaps T20 internationals need a complete rebrand. May I
propose we rename them Ecky Thump. The term was Introduced in Kung Fu Kapers, a 1975 episode of BBC comedy The Goodies.
</item 3>

<item 4>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
In the episode, ‘ecky thump’ is a secret Lancastrian martial art using black puddings as a weapon. I’m not suggesting we go as far introducing black puddings
immediately. A phased withdrawal of the cricket bat will cause less fuss.” But what would you call The Hundred? Bloody Eck?

Updated at 22.00 GMT

6d ago 21.54 GMT WICKET! West Indies 20-1 (King c Buttler b Topley 3) One down, nine to go. Brandon King slogs Topley miles in the air towards mid-on, where
Buttler takes a simple catch. Jos Buttler takes the catch to dismiss Brandon King, and the first West Indies wicket is gone. Photograph: Randy Brooks/AFP/Getty
Images King traipses off the pitch.
</item 4>

<item 5>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 01.55 GMT 16th over: West Indies 141-6 (Powell 26, Russell 4) Livingstone continues. Four singles off the first four balls keeps the pressure on the
West Indies. HOLD THAT THOUGHT. Rovman Powell crunches consecutive sixes down the ground with a shrug and swagger to make it sixteen off the over. The scales
tip once more, England need to do away with one or both of Powell and Russell – there’s no way they don’t win this for West Indies if they are there at the end.
31 needed from 24 balls.
</item 5>

<item 6>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
That’s Rashid’s 99th T20I wicket – in truth it was a bit of a drag down but from the front of the hand so it skidded onto Mayers who didn’t time it and was
caught by Brook in the deep. England needed that BIG TIME. A masterclass from Rashid in the rest of the over, Nicholas Pooran is the new man and his clip for
one is the only run from the over. 8th over: West Indies 79-2 (Hope 19, Pooran 1)

Updated at 00.56 GMT

13 Dec 2023 00.28 GMT 7th over: West Indies 78-1 (Mayers 35, Hope 19) Rehan to Kyle Mayers. First ball is dragged down and launched over the midwicket boundary
for SIX. Pressure on the young leggie. What a shot! Mayers holds the pose as he lofts a supreme drive down the ground for SIX more. Brutal elegance.
</item 6>

<item 7>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.03 GMT 2nd over: West Indies 30-0 (King 21, Mayers 9) Will Jacks to bowl his workmanlike spin at the other end. Uh-Oh! He drops short and Kyle
Mayers murders it, the ball lands on the roof of the stadium. That deserved everything it got. Mayers cuts for three behind point and then King skips jauntily
out of his crease to skim the ball for four down the ground!



12 Dec 2023 23.59 GMT 1st over: West Indies 16-0 (King 16, Mayers) Brandon King gets the home side off to a blistering start, sending Sam Curran down the ground
for four and swiping a brace of SIXES over mid-wicket. Game on!



12 Dec 2023 23.56 GMT No sooner have I sploshed some water on me teabag… the players emerge. West Indies need 172 to win!
</item 7>

<item 8>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 21.31 GMT 19th over: England 129-8 (Curran 9, Rashid 1) The hat-trick ball is another attempted yorker that Rashid flicks wide of Russell for a single.
He would have had three wickets in four balls had Hosein taken a tricky low chance at third man to dismiss Curran. Russell, who bowled really well, finishes
with two for 25.
</item 8>

<item 9>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 02.36 GMT James Wallace That’s it from me this evening/morning. Time to haul myself out of the OBO armchair and skulk off for some kip. That was an
intriguing game and sets up this T20I series nicely. A real statement has been laid down by Rovman Powell and his men, West Indies were up against it early
doors with England crunching their way to 112-2 off the first ten overs but Jos Buttler’s side only managed to make 59 runs off the next ten, losing eight
wickets along the way. A total of 171 was thirty or more shy of where England would have liked to have ended up and so it proved as West Indies peeled up their
sleeves and showed how muscular their own batting card is – the home side cleared the ropes fourteen times compared to England’s six. Andre Russell and Rovman
Powell found the stands with ease at the business end of the match to get West Indies over the line with eleven balls remaining.
</item 9>

<item 10>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Just as surely as when they came out on Tuesday facing an almost unachievable task, the outcome of the West Indies’ run chase seemed inevitable from the start.
With such a meagre target they needed little more than to keep their heads, and though both Topley and Sam Curran conceded only two with their final overs to
push the contest into its final over, that never looked in doubt.

skip past newsletter promotion Sign up to The Spin Free weekly newsletter Subscribe to our cricket newsletter for our writers' thoughts on the biggest stories
and a review of the week’s action Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more
information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy
. We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Quick Guide How do I sign up for sport breaking news alerts?
</item 10>

<item 11>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Photograph: Randy Brooks/AFP/Getty Images King traipses off the pitch. Photograph: Randy Brooks/AFP/Getty Images

Updated at 22.14 GMT

6d ago 21.54 GMT 2nd over: West Indies 20-0 (King 3, Charles 13) Johnson Charles was playher of the match when West Indies beat England at the T20 World Cup 11
years ago. Charles walloped 84 off 56 balls that day, and he looks like he fancies something similar here. He pulls Woakes for a big six, albeit with a hint of
top-edge, and glides a boundary past short third man. West Indies are off to a flyer.
</item 11>

<item 12>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
Russell at his best adds so much to West Indies's side. With him at 7, and Holder, Shepherd and Hosein to follow, West Indies bat incredibly deep — Tim Wigmore
(@timwig) December 12, 2023

Updated at 23.48 GMT

12 Dec 2023 23.46 GMT England all out 171! Tymal Mills gets a gossamer thin edge through to Pooran and England are all out with three deliveries to spare! What
a turnaround for West Indies who were staring down the barrel with England flying 77-0 after six overs.



12 Dec 2023 23.44 GMT WICKET! Rashid b Joseph 0 (England 170-9) Rashid goes to swipe Joseph in the final over and hacks the ball onto his timbers. England nine
down…
</item 12>

<item 13>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
12.4 overs: West Indies 108-4 (Hope 27, Powell 6)

Updated at 01.09 GMT

13 Dec 2023 00.58 GMT 12th over: West Indies 102-4 (Hope 26, Hetmyer 1) Just a single and the wicket off the over, Shai Hope can’t break the legspin shackles.
Windies skipper Rovman Powell is the new man and Buttler cranks the pressure by inserting a slip.

Updated at 01.04 GMT

13 Dec 2023 00.52 GMT WICKET! Hetmyer c Duckett b Rashid 1 (West Indies 101-4) Adil Rashid takes his 100th T20I wicket! A crucial one in the context of the game
too – the dangerous Hetmyer is sent on his way for just a single as England look to ramp up the pressure heading into the nub end of this match.
</item 13>

<item 14>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
We’ll be back to cover the second match in Grenada on Thursday, do join us for that. Until then, thank you and goodnight.

Updated at 02.36 GMT

13 Dec 2023 02.28 GMT Andre Russell is Player of the Match: In his first T20I in over two years he took 3-29 and smashed 29* off just 14 deliveries with the
game in the balance. He seems very pleased to be back in maroon. Life is so funny. Since when I got selected for the West Indies two weeks ago I’ve been
dreaming of being man of the match.”
</item 14>

<item 15>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
That is all.”



6d ago 22.13 GMT 6th over: West Indies 39-2 (Charles 16, Hope 5) Charles mistimes a short ball from Woakes just short of Moeen in the covers. In the end Moeen
does well to save the boundary. This has been a good fightback by England, with only nine runs scored in the last three overs.



6d ago 22.09 GMT 5th over: West Indies 37-2 (Charles 15, Hope 4) Topley continues, which makes sense because the spinners will be doing most of the work after
the Powerplay, and concedes only three runs. He has quietly had a good series and is comfortably the most economical seamer on either side.
</item 15>

<item 16>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 01.45 GMT WICKET! Shai Hope c Brook b Rehan Ahmed 36 (West Indies 123-5) Ahmed is bunted for SIX by Shai Hope but strikes with the next ball as Hope
holes out!



13 Dec 2023 01.44 GMT 14th over: West Indies 112-4 (Hope 27, Powell 10) Liam Livingstone instead of Rashid and he does well to keep it to singles, England
trying to take this as deep as possible and hope West Indies fluff their lines at the death. West Indies need 55 from 36 balls.
</item 16>

<item 17>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.10 GMT 3rd over: West Indies 36-1 (Mayers 10, Hope 4) Shai Hope is the new man and digs out a Woakes yorker and somehow manages to crunch it away
down the ground for four. The replay of the Duckett catch is extremely enjoyable, Woakes owes him one for that as it was a poor ball outside off stump.



13 Dec 2023 00.07 GMT WICKET! Brandon King c Duckett b Woakes 22 (West Indies 32-1) Chris Woakes into the attack to replace the beleaguered Sam Curran. OH. MY.
LORDY. Ben Duckett flies to his left at backward point to pull off a spectacular diving catch off a full blooded cut shot from King. Stunner!
</item 17>

<item 18>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.37 GMT 12th over: West Indies 80-3 (Hope 15, Rutherford 20) England appeal unsuccessfully for caught behind when Rutherford misses a swipe at Rashid.
There’s only one review left and they decide not to risk it. A good job too; there was nothing on UltraEdge. Out of nothing Rutherford launches a full ball from
Rashid down the ground for six, almost knocking himself off his feet in the process. West Indies are managing an awkward chase pretty well.
</item 18>

<item 19>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 21.39 GMT The difference two days make Tuesday, Tarouba England 267-3, 19 sixes

Thursday, Taroura England 132 all out, five sixes

Updated at 21.45 GMT

6d ago 21.36 GMT WICKET! England 132 all out (Curran c Rutherford b Holder 12) Sam Curran clouts the next ball to long-off to end an increasingly miserable
England innings. They were 109 for four after 14 overs, and then the house fell down. West Indies need 133 to win the series.

Updated at 21.38 GMT

6d ago 21.33 GMT WICKET! England 132-9 (Rashid run out 1) Adil Rashid sacrifices himself, just like in the 2019 World Cup final, so that Curran can stay on
strike. Four balls remaining.
</item 19>

<item 20>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 01.41 GMT 13th over: West Indies 112-4 (Hope 27, Powell 10) Rehan has two balls left to complete his over. The first ball is flayed for four by
Powell behind point! Valuable runs for West Indies. Rehan’s final ball is a dot, defended by Powell into the off side. West Indies need 60 from 42 balls.

Updated at 01.42 GMT

13 Dec 2023 01.30 GMT Simon Burnton is our man in Barbados and he sends good news! “It’s a 9.35pm restart. We haven’t lost any overs.” We should be underway in
about eight minutes. I believe that 10.10pm local time is the cut off for the match – that’ll be 2.10am here in London town. Excuse me whilst I head to the
kitchen to microdose some Kenco.
</item 20>

<item 21>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
It would have been four but for an outstanding stop from Rehan Ahmed.



6d ago 23.12 GMT 19th over; West Indies 124-6 (Hope 37, Holder 1) Holder is almost run out first ball! He took a very dodgy single to mid-off and was well short
of the crease when Curran just failed to gather Rashid’s throw. Nine to win from six balls, Chris Woakes to bowl them.



6d ago 23.10 GMT WICKET! West Indies 123-6 (Russell c Livingstone b Curran 3) Oh my. Russell slugs a low full toss straight down the throat of Livingstone at
long-on, and England – who have been behind the game throughout this innings – are again dreaming of an improbable victory. West Indies need 10 from 7 balls.
</item 21>

<item 22>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 23.00 GMT 17th over: West Indies 114-5 (Hope 30, Russell 1) The wicket England really need is that of Shai Hope, who is quietly guiding West Indies
towards the finish line.
</item 22>

<item 23>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 02.09 GMT West Indies win by 4 wickets! Andre Russell smashes the winning runs off Sam Curran! A length ball is spanked over point for a one bounce
four to give the home side the win, they go 1-0 up in the series.

Updated at 02.29 GMT

13 Dec 2023 02.07 GMT 18th over: West Indies 168-6 (Powell 31, Russell 25) Adil Rashid is recalled for his final over, can he turn the game back to England?
Nope. ANDRE RUSSELL SMASHES THE RASHID GOOGLY CLEAN OUT OF THE GROUND! Huge SIX! The big man even allows himself a coy smile. He enjoyed that one. A dot the
next ball is followed by a powerful drive all along the baize for four! Eleven in total off the over and West Indies now just need four runs.
</item 23>

<item 24>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.05 GMT 4th over: West Indies 34-2 (Charles 15, Hope 1) England needed early wickets to have a realistic chance of victory. One more in the Powerplay,
before they bring on Adil Rashid, would put them right back in the game.



6d ago 22.03 GMT WICKET! West Indies 33-2 (Pooran b Woakes 10) UltraEdge shows Pooran did edge that ball from Topley – but it doesn’t matter because Woakes has
got him! Pooran tried to ride the bounce of an awkward delivery that cramped him for room, took the inside of an angled bat and ricocheted onto the stumps.
Nicholas Pooran looks glum as his bails go flying.
</item 24>

<item 25>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.16 GMT 4th over: West Indies 47-1 (Mayers 18, Hope 5) Tymal Mills comes onto bowl for England for the first time in eighteen months, he has
plenty of T20 skills up his sleeve, can he showcase them here? Hmmm, a couple of wides slanted down the leg side isn’t the best of starts. Mills lets out a
hearty groan at the second. Two full balls are worked for singles… Kablammo! Mills to Mayers – short ball – pulled onto the roof for SIX! What a shot, new ball
please! DROP! And it is Ben Duckett who shells it! It would have been even better than the first grab but the ball smashes into his palm as he leaps to his
right and comes straight out. They stick or they don’t. It’s all happening.
</item 25>

<item 26>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.37 GMT 9th over: West Indies 87-2 (Hope 23, Pooran 3) Will the wickets slow the Windies as they did England? Rehan is replaced by Tymal Mills who
starts with another wide. Rusty stuff so far from the southpaw seamer. That’s better – a series of slower balls and back of the hand trickery restricts Pooran
and Hope before the final ball is top edged over Buttler’s head for a one bounce four. West Indies need 85 from 66 balls.



13 Dec 2023 00.32 GMT WICKET! Kyle Mayers c Brook b Rashid 35 (West Indies 79-3) The sorcerer follows his apprentice and bags a wicket first ball!
</item 26>

<item 27>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 02.01 GMT 17th over: West Indies 157-6 (Powell 31, Russell 14) Tymal Mills is entrusted. Singles off the first two balls. Powell swipes a full ball
into the deep to collect a couple. SIX! Mills goes short and Powell gets a mighty top edge for a maximum! Fortune favours the brave, the Windies skipper gave
that everything and it could have gone anywhere. Mills sends one down the leg side – wide ball. Pressure on the bowler here… Shot! Short and wide from Mills and
sent to the fence with Swiss clock timing by Powell. Sixteen off the over, looking terminal for Buttler and co. 15 needed from 18 balls.
</item 27>

<item 28>
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
</item 28>

<item 29>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Adil Rashid (right) celebrates the dismissal of Johnson Charles (left). Photograph: Randy Brooks/AFP/Getty Images

Updated at 22.43 GMT

6d ago 22.17 GMT 7th over: West Indies 51-2 (Charles 26, Hope 7) Rehan Ahmed’s first over disappears for 12. It started quite gently, with four singles and a
two off the first five balls, but then Charles clattered a no-nonsense straight six. “We’re not going to win this, Rob,” writes Guy Hornsby. “But I’m just here
to say I’m still watching for Woakes’ seam position and Rehan’s googly. That is all.”
</item 29>

<item 30>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 23.09 GMT 18.4 overs: West Indies 123-5 (Hope 37, Russell 3) Sam Curran concedes only one run from the first four balls of a superb penultimate over. The
ball is getting very wet, though, so there’s a short break in play while England call for a dry towel. West Indies need 10 from 8 balls.



6d ago 23.04 GMT 18th over: West Indies 122-5 (Hope 36, Russell 3) Jos Buttler gambles by continuing with Rehan Ahmed, who has been relatively expensive. It
doesn’t come off: Hope skims a back cut between short third and backward point for four to move the requirement back below a run a ball. It’s very hard to see
how England can with this now.
</item 30>

<item 31>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Amid the wider theme of transformation Adil Rashid’s excellence remains constant. The 35-year-old, who rose to the top of the T20 bowling rankings for the first
time on Wednesday, took two wickets and conceded just 21 while Topley was even better, also claiming a pair of victims but conceding only 17. England take
plenty of positives from this series, but the result was not one.
</item 31>

<item 32>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
Brutal elegance. Rehan comes back well with two sharply turning leggies. Bosh! Shai Hope gets in on the act and smears a length ball over mid on for the third
six of the over. It’s a learning curve for the tyro spinner but that doesn’t make it any less bruising.



13 Dec 2023 00.23 GMT 6th over: West Indies 59-1 (Mayers 22, Hope 13) Curran is called back, Mills granted just the one over for the time being. Two slower
balls bring two soothing dots for England. A single each to Mayers and Hope. Curran does very well to get out of the over with just three runs conceded.
Powerplay complete. How long till we see some spin from Rehan? Not long at all, he’s on for the next over.
</item 32>

<item 33>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
West Indies are off to a flyer.



6d ago 21.49 GMT 1st over: West Indies 7-0 (King 1, Charles 2) There’s some early inswing to the right-handers for Reece Topley. Brandon King survives an LBW
appeal first ball – it was missing leg – before another inswinger to Johnson Charles runs away for four leg-byes. Every boundary hurts when the target is so
low, even in the first over.



6d ago 21.45 GMT After a very short break, the players are back on the field.
</item 33>

<item 34>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Photograph: Ricardo Mazalán/AP 𝗔 𝗦𝗜𝗫 𝗧𝗢 𝗙𝗜𝗡𝗜𝗦𝗛! 🙌@WindiesCricket claim the series 3-2 and a white-ball clean-sweep 🏏#WIvENG pic.twitter.com/chrYkHbtcp — Cricket
on TNT Sports (@cricketontnt) December 21, 2023

Updated at 23.34 GMT

6d ago 23.13 GMT 19.1 overs: West Indies 127-6 (Hope 37, Holder 4) Holder inside edges Woakes this far wide of leg stump for three runs. It would have been four
but for an outstanding stop from Rehan Ahmed.
</item 34>

<item 35>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.45 GMT 14th over: West Indies 94-3 (Hope 19, Rutherford 30) West Indies are rushing towards a series victory. Rutherford hooks the new bowler Sam
Curran for six more, and there are 11 from the over in total. Rutherford has 30 from 23 balls, Hope 19 from 28. They’re playing their roles to perfection.



6d ago 22.40 GMT 13th over: West Indies 83-3 (Hope 17, Rutherford 21) Rehan Ahmed returns, perhaps in the hope of tempting Rutherford into one big shot too
many. Instead it’s Shai Hope, who is happily playing the anchor role, who faces five of the six deliveries. England are running out of time.
</item 35>

<item 36>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
Hetmyer danced out of his crease and got a good piece of the length ball but it flew flat and hard to Duckett on the midwicket sponge. And he does it on his
100th T20I appearance! 💯#BBCCricket #WIvENG pic.twitter.com/LYhUBtNnUj — Test Match Special (@bbctms) December 13, 2023

Updated at 01.02 GMT

13 Dec 2023 00.49 GMT WICKET! Pooran c & b Rehan Ahmed 13 (West Indies 100-3) Rehan back into the attack after his first over was minced for n n n n nineteen.
Got him! Pooran flicks back a full ball and Rehan takes a nifty catch tumbling away to his left. Shimron Hetymer is the new batter and he tucks a single into
the leg side to open his account.
</item 36>

<item 37>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.57 GMT WICKET! West Indies 113-5 (Powell c Rashid b Topley 8) England haven’t given this up. Powell edges Topley to short third man, where Rashid
takes a smart low catch to his left. West Indies need 20 from 21 balls. Reece Topley (right) celebrates the dismissal of Rovman Powell. Photograph: Randy
Brooks/AFP/Getty Images

Updated at 23.07 GMT

6d ago 22.54 GMT 16th over: West Indies 112-4 (Hope 29, Powell 8) Hope, who is playing an exemplary innings, opens the face to steer Rehan for his first
boundary. Rehan has struggled as the series has progressed, an d later in the over Powell muscles him over long-on for six. West Indies are 21 runs away from a
hat-trick of T20 series wins this year: South Africa away, India and England at home.
</item 37>

<item 38>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
A wicket once so true turned untrustworthy and they struggled to a score of 132, less than half the 267 to which they had joyously motored just 48 hours
previously. And though the home side’s response was anything but overwhelming, neither was the result ever in significant doubt.

“We’ve had some high-scoring games, but the last game was almost the complete opposite of what happened here,” Reece Topley said. “We were probably just under
par with the first-innings score but with the ball we were excellent, pushed them right to the end. I think it’s been an amazing series. Both teams have played
some unbelievable cricket. I was so excited turning up today, because it was basically a final and those are the games you want to play in and be on the right
side of. The bottom line is you want to win this series, I want to win every series I can for England.”
</item 38>

<item 39>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
That made it 70 for four and the game was one ball into its ninth over, at which point Liam Livingstone and Moeen Ali had to focus for a while on disaster
prevention rather than entertainment.

Gudakesh Motie is embraced his his West Indies teammates after dismissing England’s in-form Phil Salt. Photograph: Randy Brooks/AFP/Getty Images

Suddenly bowlers were energised by the prospect of posting, for a change, less-than-humiliating figures. Akeal Hosein’s were outstanding, his four overs going
for just 20 and bringing the wickets of Will Jacks and Moeen, undone by a relay catch on the boundary after combining with Livingstone to add 40 for the fifth
wicket. Livingstone followed precisely two overs later, prodding the ball limply back into the hands of Motie, at which point things really fell apart, the last
five wickets surrendered in 20 balls for just 11 runs.
</item 39>

<item 40>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Often there is a sense as a tour enters its final days of thoughts turning to home, and perhaps the focus was not as laser-sharp as it might have been for this
game, determination less than absolute to grasp this opportunity to end England’s run of reverses in the Caribbean. Whatever the reason, the pyrotechnics that
rocketed England to victory in the two previous matches never sparked in this one. It took a fabulous delivery from Gudakesh Motie to end Phil Salt’s run of
absurd form with his score a modest 38, but some of the other wickets that fell during the period that defined England’s innings and the game seemed a little
careless.

In the fourth over Jos Buttler casually turned a Jason Holder delivery to fine leg, where Oshane Thomas was positioned, hands cupped. Harry Brook tried to
paddle-sweep Motie but sent the ball only a few feet into the air, giving Nicholas Pooran an easy catch.
</item 40>

<item 41>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.33 GMT 11th over: West Indies 71-3 (Hope 14, Rutherford 12) After 23 balls without a West Indian boundary, Rutherford lifts Moeen handsomely over
extra cover for four. That’s an excellent shot which keeps West Indies in control of the run-chase.

Updated at 22.33 GMT

6d ago 22.28 GMT 10th over: West Indies 62-3 (Hope 12, Rutherford 5) West Indies know how dangerous Rashid is and are content to sit on him. I think that was
Charles’ plan as well, but then he saw a low full toss and his brain went rogue. Time for drinks, after which West Indies need 71 from 60 balls.



6d ago 22.25 GMT Rutherford is not out! I need someone else’s glasses: it was comfortably missing leg stump and England have lost a review.
</item 41>

<item 42>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.50 GMT 15th over: West Indies 99-4 (Hope 23, Powell 1) Rashid finishes with figures of 4-0-21-2. He’s had a great series: eight wickets at 15 with an
absurd economy rate of 6.26. But he’s going to end on the losing side.



6d ago 22.48 GMT WICKET! West Indies 95-4 (Rutherford c Curran b Rashid 30) Buttler brings back Rashid for his final over, knowing it’s now or never for
England, and gets an immediate reward. Rutherford smashes a drive towards short extra, where Curran dives to his left to take a superb catch. It’s not the
despair, Laura…
</item 42>

<item 43>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.25 GMT ENGLAND REVIEW FOR LBW AGAINST RUTHERFORD! This looks really close. Rutherford pushed around a legbreak from Rashid and was hit on the pad in
front of middle and leg.



6d ago 22.24 GMT 9th over: West Indies 58-3 (Hope 11, Rutherford 2) Moeen replaces Rehan and rushes through a pretty good over, four singles from it. West
Indies are still on top but another wicket would make things very interesting.



6d ago 22.21 GMT WICKET! West Indies 54-3 (Charles c Buttler b Rashid 27) Adil Rashid strikes straight away. On this occasion it had nothing to do with his
bowling, but maybe a little to do with his aura. He accidentally bowled a very wide, dipping full toss to Charles, who clunked it low to Jos Buttler at short
extra cover.
</item 43>

<item 44>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
12 Dec 2023 23.48 GMT An innings of two halves for England From 77 for no loss to 171 all out with three balls to spare. England's recent white ball approach is
a variable one yielding mixed...often frustrating results #WIvENG — Aatif Nawaz (@AatifNawaz) December 12, 2023



12 Dec 2023 23.47 GMT Time to stick the kettle on and raid the fridge before West Indies start their chase. 3-19 for Andre Russell on his comeback for West
Indies, balancing the attack perfectly.
</item 44>

<item 45>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
The wicket has given Ahmed a confidence boost and he gets out of the over with three dots. 11th over: West Indies 101-3 (Hope 25, Hetmyer 1)



13 Dec 2023 00.43 GMT 10th over: West Indies 99-2 (Hope 25, Pooran 13) Rashid is given another after his excellent first over. The googly is spotted by Pooran
and driven down the ground handsomely for SIX! A glide behind point bring a single, Hope returns the favour with a flick into the leg side. A short ball is
panned into the deep by Pooran for a couple to end the over. Bon Jovi O’clock – West Indies need 73 from 60. Time for a quick slurp for the players and a
knuckle stretch for me. Is there anybody alive out there?!
</item 45>

<item 46>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 23.36 GMT Jos Buttler’s reaction We probably didn’t adapt quickly enough with the bat – I thought 16o w0uld have been a decent score on a used wicket. It
was a fantastic bowling effort; we did well to hang in the game as long as we did. Phil Salt has had a brilliant series. We found out lots about ourselves as a
team. [On the stop-clock experiment] It was only in the last game that we really noticed it. It’s good to keep the game moving.



6d ago 23.19 GMT England’s miserable white-ball year ends with another series defeat. Plenty of good has come from this series, though, most notably the
performance of Phil Salt. While there are a few slots to fill, most notably at No3, they are in reasonable shape ahead of nexr year’s World Cup.
</item 46>

<item 47>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
12 Dec 2023 23.42 GMT WICKET! Ahmed c Powell b Russell 1 (England 169-8) Rehan Ahmed follows shortly after, trying to smear a length ball down the ground he can
only find Powell lurking ten yards in from the sponge. Russell gets two in the over and finishes his spell and his first T20I in two years with figures of 3-19.
Nicely done. 19th over: England 169-8 (Woakes 2, Rashid 0)



12 Dec 2023 23.37 GMT WICKET! Livingstone b Russell 27 (England 167-7) Livingstone chops a cutter from Russell onto his stumps!
</item 47>

<item 48>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Updated at 23.20 GMT

6d ago 23.16 GMT I suppose it’s appropriate that it ended with a six, the 120th of the series. Hope played a gem of an innings there, calmly anchoring West
Indies in an increasingly nervous chase with a run-a-ball 43. England fought admirably to defend a below-par total; but for Hope, they would probably have done
it.

Updated at 23.17 GMT

6d ago 23.14 GMT WEST INDIES WIN THE SERIES 3-2! 19.2 overs: West Indies 133-6 (Hope 43, Holder 4) That’s it! Hope pings Woakes over extra cover for six to seal
another series victory for West Indies! West Indies’ Shai Hope (right) celebrates with Jason Holder after hitting a six to win the match by 4 wickets.
</item 48>

<item 49>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
What a difference two days make. Somehow Tuesday’s ludicrous, record-smashing, high-scoring entertainment was followed by a sequel on the same surface that
eventually delivered thrilling low-scoring entertainment, and a different result claimed in very different style. It was eventually won for West Indies with
four balls to spare and in the most unfitting of ways – with a massive six – by Shai Hope, again proving himself the right man for a run-chase.

So a series marked by stunning power hitting, roaring run rates and an unexpected midway transformation in England’s performances and prospects concluded with a
3-2 triumph for West Indies, who extended their supremacy over England at home to four series in three formats across nearly five years.

England assembled exactly the same players in exactly the same place and exactly the same situation – losing the toss, put in to bat – but this was an entirely
different performance.
</item 49>

<item 50>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.20 GMT 5th over: West Indies 56-1 (Mayers 20, Hope 12) Woakes continues, after five balls he’s gone for just three runs. Can he get out of the
over cheaply? No – A length ball is clubbed down the ground for SIX with a minimum of fuss by Shai Hope. West Indies have come out of the traps as well as
England did, one over of the Powerplay left.
</item 50>
=== END ITEM id=e13478200002b51f ===

=== ITEM id=2ce36df98acbc608 ===
Gold sentences and facts:
Sentence 0: Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence.
  Fact 0.0: Google's public documentation says Google uses three categories to rank results that are local to a searcher
  Fact 0.1: The three categories are relevance, distance, and prominence
Sentence 1: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 1.0: The last roundup reported that Google spent $26.3 billion in 2021 making Google the default search engine across platforms
  Fact 1.1: The last roundup reported that Google tried to have Chrome preinstalled on iPhones
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: A case was filed by Arkansas-based publisher Helena World Chronicle
  Fact 2.1: The case argues that Google "siphons off" news publishers' content
  Fact 2.2: The case argues that Google siphons off news publishers' readers
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue
  Fact 2.4: The case argues that Google does this through anticompetitive means

Retrieved items:
<item 1>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
These are exactly the arguments that other search engine companies, like DuckDuckGo, Brave or Microsoft’s Bing, make when they accuse Google of being
anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying Google has become the monopoly it denounced years ago.

What does it all mean?

The case is expected to continue for several weeks, bringing to a head one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the
1990s.

If the judge rules against Google, the outcome could look a lot like the Microsoft deal, in which the computer company was required to change its behavior and
share its APIs with third-party developers. Microsoft was also banned from making anticompetitive and exclusive deals with computer manufacturers.

Google might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more
users. The DOJ has said that Google gets 16 times more data than Bing does everyday.
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 2>

<item 3>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
That doesn’t make its actions illegal, the company says.

Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in
2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.

Google paid $26 billion in 2021 to be everyone’s default search engine

When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple
browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.

About $18 billion of that total amount went directly to Apple, according to a New York Times report published earlier this month. Google has had its deal with
Apple in place since 2003.

When questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind.
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 4>

<item 5>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Didn’t other parties sue Google too?

They did! And then, they all settled. All 50 state attorneys general and a number of consumers have tentatively settled with Google over app store antitrust
claims, though it’s not clear what the deal might include.

Match Group, the company behind Tinder, Match.com, OkCupid and other dating apps, just reached a surprise last-minute settlement too, one that makes it sound
like Match effectively gave up.

Epic stands alone.

Could we get a settlement between Google and Epic as well?

Epic seems to have no interest in settling as long as the “Google tax” exists, and the company’s repeatedly said it’s not interested in cash. So far, the
company’s been willing to push these cases all the way through multiple appeals, cost be damned.

It’s not clear what Google could proactively offer that would satisfy Epic, since CEO Tim Sweeney has also publicly suggested that Google would simply
circumvent any settlement it offers.
</item 5>

<item 6>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 7>

<item 8>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 9>

<item 10>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 10>

<item 11>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Legal cases build and break

We saw a fair number of lawsuits filed in 2023, but few saw any real movement, let alone success. Most suits over copyright and other missteps in the AI
industry are still pending. 2024 will see a lot of them fall by the wayside, as companies stonewall critical information like training data and methods, making
allegations like the use of thousands of copyrighted books difficult to prove in court.

This was only the beginning, however, and many of these lawsuits were filed essentially on principle. Though they may not succeed, they may crack the process
open far enough during testimony and discovery that companies would rather settle than have certain information come to light. 2024 will bring new lawsuits as
well, ones pertaining to misuse and abuse of AI, such as wrongful termination, bias in hiring and lending, and other areas where AI is being put to work without
a lot of thought.
</item 11>

<item 12>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 12>

<item 13>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
</item 13>

<item 14>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.
</item 14>

<item 15>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 15>

<item 16>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google CEO Sundar Pichai testified Monday in the U.S. government’s antitrust trial against the company. The executive defended Google’s business tactics,
including its deal with Apple and other partners to make Google the default search engine.

The lawsuit stems from a 2020 antitrust claim over Google’s dominance in the online search market. The claim is separate from one filed in January by the
Department of Justice, along with eight states including New York, California and Colorado, that aims to “halt Google’s anticompetitive scheme, unwind Google’s
monopolistic grip on the market and restore competition to digital advertising.”

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products.

Google argues that it’s just doing business. Everybody wants Google as the default engine because it’s the best. That doesn’t make its actions illegal, the
company says.
</item 16>

<item 17>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 17>

<item 18>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The amount that Google spent on homepage market share has been a fixing point in the trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified
that their search engines would have been far more successful, even competitive with Google, had they been able to make similar deals with Apple. Microsoft CEO
Satya Nadella even said he was willing to spend $15 billion per year to get Bing into Apple’s default search, per The Information.

Google agreed not to promote Chrome to Safari users

As part of its search deal with Apple, Google agreed not to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with
banners, pop-ups and other annoying means in other Google apps.

The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.

Google tried to be preinstalled on iPhones

Pichai admitted to attempting to get Tim Cook to preinstall Google on every iOS device back in 2018, according to The Verge.
</item 18>

<item 19>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 19>

<item 20>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel contains a summary of
content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information from Publishers’
websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.

By 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was
content “misappropriated from Publishers,” the complaint alleges.

Other Google technologies, like “Featured Snippets” where Google algorithmically extracts answers from webpages, were also cited as shifting traffic away from
publishers’ websites.

More importantly, perhaps, is the suit’s tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The
Wall Street Journal, which led with a shocking statistic.
</item 20>

<item 21>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
The RSAs also motivate other companies to make or sell more Android devices, he said.

Expedia complains of too many ads on search, expensive ad payments

On November 1, Barry Diller, chair of Expedia and IAC, testified about his concerns regarding the increasing number of ads in search results having an impact on
organic listings.

“I must say I’m on the edge of revolt now that Google’s actions are so punitive, not just for Expedia but also for IAC and all the players that depend upon
something of a level playing field,” wrote Diller in a letter to Google back in 2019, according to Bloomberg.

Google CEO Sundar Pichai fired back that Google’s travel listings was one of the most popular experiences the company built.

Expedia execs also testified about the cost of ads and how increases had no impact on search results.
</item 21>

<item 22>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 22>

<item 23>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 23>

<item 24>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Google spent its 45 minutes attempting to explain that its dominance over the Android app market isn’t anything nefarious but simply the natural outcome of
Google fiercely competing with the iPhone and its iOS App Store, where Google would like the court to believe that competition truly lies.

If Google can convince the jury of that, it could be a winning argument in the case — because obviously, Google doesn’t have a monopoly on app stores or phones
in general. “You cannot separate the quality of a phone from the quality of the apps in its app store, and that means Google and Apple compete against each
other,” began Google lead attorney Glenn Pomerantz.

But Google wound up spending much of its opening statement attempting to explain away its seemingly bad behavior as normal business practices and didn’t always
succeed out of the gate.
</item 24>

<item 25>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.
</item 25>

<item 26>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 26>

<item 27>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 27>

<item 28>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
He hoped to make Google and Apple’s services so connected as to be inseparable.

The way Pichai pitched it would have been a win-win for both companies. Google gets more people searching on its platform — not to mention all that juicy data —
and Apple would get more revenue, as a result of the lucrative search agreement the two signed.

For whatever reason, Cook didn’t take the bait. Apple doesn’t preload third-party software onto its devices, and it wasn’t going to make an exception for
Google.

Google’s deleted chat logs

During Pichai’s testimony, the DOJ touched on Google’s policy of deleting internal chat messages, despite being subject to a litigation hold. In February, the
DOJ accused Google of systematically destroying chats through its history-off option, which deletes messages every 24 hours unless a user manually changed the
setting.

Pichai said that he took action against the history-off default for chat in February to comply with the DOJ’s litigation hold, according to CNBC.
</item 28>

<item 29>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 29>

<item 30>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 30>
=== END ITEM id=2ce36df98acbc608 ===

=== ITEM id=3fa6d24c718ac0ec ===
Gold sentences and facts:
Sentence 0: Shares of Rogers extended losses on news of the lawsuit, closing down 2.2% to C$53.07 in Toronto.
  Fact 0.0: Shares of Rogers extended losses on news of the lawsuit.
  Fact 0.1: Rogers shares closed down 2.2% to C$53.07 in Toronto.
Sentence 1: Amazon tumbled 4 per cent after the Federal Trade Commission and 17 state attorneys general filed an antitrust lawsuit against it.
  Fact 1.0: Amazon tumbled 4 per cent after the Federal Trade Commission filed an antitrust lawsuit against it.
  Fact 1.1: 17 state attorneys general filed an antitrust lawsuit against Amazon.
  Fact 1.2: The Federal Trade Commission and the 17 state attorneys general filed the antitrust lawsuit against Amazon together.

Retrieved items:
<item 1>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“We can tailor our ads to hit exactly the consumers we want,” Moller said. “It’s a perfect marriage of a transaction, and that’s one of the beauties of what
their marketplace offers.” Needham said he feels he would have been more supportive of the case if it were filed a few years ago, pre-pandemic. At that time, he
said, "I would have felt, yes Amazon is a monopoly… But actually after Covid, into 2023, ecommerce has had a lot of big changes.” He added, "The competition is
just not what it was in 2019." Competitors like Shopify and Walmart are increasingly viable alternatives for many categories of sellers, Needham said, not to
mention rapidly growing Chinese e=commerce companies like Temu. As a result, Needham said he’s seen some significant changes from Amazon. Among those is a
greater ability for Amazon sellers to communicate with buyers, offering select customers certain promotions.
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 2>

<item 3>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The Shaw takeover finally closed on April 3 of this year, and the Rogers siblings engaged in talks “intended to resolve all issues between us,” Rogers-Hixon
stated in an affidavit. “I was hopeful all issues could be resolved without negative publicity.” But those negotiations collapsed last month, she said.

It isn’t clear from the court filings whether the discussions included the possibility that any Rogers family members would sell their stake in Rogers
Communications, leave their roles at the company, or exit the family trust that controls almost all of the voting shares of the public company.

“This matter should be resolved privately,” Rogers Communications spokesperson Sarah Schmidt said by email. “We’ve demonstrated that we will not be distracted
by these actions — we have significant momentum in the market, our merger is tracking ahead of plan, and we remain squarely focused on doing what’s right for
our customers and stakeholders.”
</item 3>

<item 4>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Hart said Amazon sales comprise about 90% of his business, meaning any hit those sales take on Amazon has a considerable impact. He recalled “24 anxious hours”
in September 2022 when a third-party seller of his popular construction toy Brain Flakes listed the toy for a lower price on Target than it was offered on
Amazon.

Molson Hart, CEO of Viahart, an educational toy company that sells on Amazon. Courtesy: Molson Hart
</item 4>

<item 5>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
A worker sorts out parcels in the outbound dock at the Amazon fulfillment center in Eastvale, California, on Aug. 31, 2021. Watchara Phomicinda | MediaNews
Group | The Riverside Press-Enterprise via Getty Images

It was late in the day on Oct. 27, 2021, when Fred Ruckel received the dreaded automated email from Amazon . Amazon’s software had detected that Ruckel’s
popular cat toy, called the Ripple Rug, was being sold somewhere else for a cheaper price. His product would no longer be shown in Amazon’s all-important buy
box, an area of the listing where shoppers click “Add to Cart.” Ruckel is the sole seller of the Ripple Rug on Amazon, so the move all but ensured his product
would disappear from the website, costing him thousands of dollars per day.
</item 5>

<item 6>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The documents show that the fractures inside the prominent Toronto family have never healed. Rogers Communications is the country’s largest wireless provider
and the owner of a cable television empire, the Toronto Blue Jays baseball team and other media and sports assets. The family’s controlling stake in the public
company is worth close to C$8 billion ($5.9 billion), according to calculations based on company filings for its 2023 annual meeting.

Shares of Rogers extended losses on news of the lawsuit, closing down 2.2% to C$53.07 in Toronto. They’ve dropped 16% this year.

The children of late founder Ted Rogers have frequently fought one another — sometimes driving down Rogers’ stock price as investors grew worried about the
company’s governance.

The bad blood spilled into full view in October 2021, when Edward Rogers publicly battled his sisters and his mother for control of the board and won in court.
The result was that five Rogers Communications directors were fired, replaced by five allies of Edward Rogers.
</item 6>

<item 7>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
The end result of these tactics, the FTC argues, is elevated prices across the web. The company steadily hikes the fees it charges sellers and prevents them
from discounting on other sites, so sellers often inflate their prices off of Amazon, creating an “artificial price floor everywhere,” according to the
complaint. The FTC is seeking to hold Amazon liable for allegedly violating anti-monopoly law, though it has not yet outlined the specific remedies it believes
would best resolve its concerns. In antitrust cases, remedies are often determined only after a court finds the defendant liable. In a blog post, Amazon general
counsel David Zapolsky said third-party sellers set their own prices on the marketplace. The company also invests in tools to help sellers offer “competitive
prices,” he said. “Even with those tools, some of the businesses selling on Amazon might still choose to set prices that aren’t competitive,” Zapolsky said.
</item 7>

<item 8>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
Still Excluded

Both Rogers-Hixon and Martha Rogers said in documents that they’re still being excluded from some parts of board and committee meetings and are being given
redacted board documents.

“I cannot properly perform my duties as a director if information that is provided to the rest of the directors is being withheld from me,” Rogers-Hixon said in
her affidavit. The two sisters are seeking a judge’s order to force the company to share information with them and allow them to attend, in full, all meetings
of the board, as well as board committees they belong to.

Rogers took “reasonable steps” to protect company information from a competitor, Schmidt said. Rogers-Hixon and Martha Rogers could have resolved the matter by
choosing a different lawyer, “and they chose not to do so,” she added.

The court filing also suggests that Rogers-Hixon was forced out of her role as a director of Maple Leaf Sports & Entertainment, the company that owns Toronto’s
professional hockey and basketball teams and in which Rogers Communications is a large minority investor.
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 9>

<item 10>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Amazon in recent years made the forum anonymous, but users must have a seller account in order to post. Another post included a screenshot of a message Amazon
sent to sellers the day after the FTC filed its complaint, which said, “As your partners, we know that this news may generate questions for you and our business
together. This lawsuit does not change anything about our relationship with you or how we operate today.” One user called it “BS verbiage,” adding, “Businesses
that sell in their store are indeed customers. And which of us has gotten good customer service?” Another user described their experience in the last 12 months
of selling on Amazon as “being up all night at an effing casino but I'm stuck, the drugs are starting to wear off, but I'm trying to break even on the mortgage
payment I'm using to play. That's how it is selling on Amazon right now to me.” The seller went on to describe the experience as a “race to the bottom.”
</item 10>

<item 11>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
The seller went on to describe the experience as a “race to the bottom.” “It's long overdue," another commenter wrote. "When they close me down, I'm applying
for a job with the FTC.” Still, others commented that the FTC’s complaint is misguided. “Selling on Amazon is a life-changing opportunity and the amount of
sellers that throw stones at the platform is astounding,” one user wrote.

Seller skepticism

Even sellers who may be sympathetic to the idea of regulating Amazon have concerns, specifically that the FTC's highlighted issues aren't necessarily ones that
would make the seller and consumer experience better. Scott Needham, who sells on Amazon and runs a product-finder tool for other Amazon sellers, said he was
“surprised by some of the points that the FTC selected.”

watch now

“I have over the years been very critical of Amazon,” Needham told CNBC. “I’ve lost a lot of sleep because of some of the things that they have done.
</item 11>

<item 12>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
Two members of the billionaire family that controls Rogers Communications Inc. are suing the company, saying they’ve been frozen out of board meetings and
blocked from getting information, reigniting a bitter power struggle within one of Canada’s wealthiest clans.

Melinda Rogers-Hixon says she’s the target of a “personal vendetta” by her older brother, Rogers Chair Edward Rogers, according to a letter contained in the
court filing. The Rogers siblings were close to an agreement that would have resolved their longstanding differences, but Edward Rogers reneged on it, according
to the Sept. 20 missive by Rogers-Hixon, which was sent to Chief Executive Officer Tony Staffieri.

The court petition was filed by Rogers-Hixon and her sister, Martha Rogers, and it argues the company has been “oppressive” to them.

The documents show that the fractures inside the prominent Toronto family have never healed.
</item 12>

<item 13>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The result was that five Rogers Communications directors were fired, replaced by five allies of Edward Rogers. Soon after, then-CEO Joe Natale was sacked and
replaced by Staffieri, the former chief financial officer. Rogers-Hixon and Martha Rogers had opposed the CEO change.

In early 2022, family members agreed to temporarily shelve their disagreements, according to this week’s court filing, to avoid negative publicity as the
company sought regulators’ approval for the takeover of Shaw Communications Inc. The C$20 billion transaction was the company’s biggest deal ever and among the
largest corporate transactions in Canadian history.

Negotiations Collapse

As part of that truce, Rogers-Hixon and Martha Rogers agreed not to participate in board discussions about the Shaw deal or receive materials on it, because of
an argument about Rogers-Hixon’s personal lawyer, whose firm also does work for Telus Corp., a competitor to Rogers and Shaw.
</item 13>

<item 14>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 14>

<item 15>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“Just like any store owner who wouldn’t want to promote a bad deal to their customers, we don’t highlight or promote offers that are not competitively priced.”
Zapolsky argued the FTC’s lawsuit could force it to stop highlighting low prices, “a perverse result that would be directly opposed to the goals of antitrust
law.”

“Long overdue” lawsuit

On Amazon’s own forum for merchants, called Seller Central, several users cheered on the FTC and said they hoped it would result in changes to the company’s
business practices. Amazon’s tense relationship with merchants has been well-chronicled over the years, with sellers expressing a range of grievances over
issues like rising fees, an arcane suspensions process, and heightened competition on the marketplace from all sides, including the e-commerce giant. “I think
it’s great, Amazon deserves it,” one person commented, adding, “More should be coming on the way.”
</item 15>

<item 16>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Fred Ruckel

Nearly three years later, Ruckel’s experience hits at the core of a sweeping antitrust lawsuit filed last week by the Federal Trade Commission against Amazon.
The agency accused Amazon of wielding its monopoly power to squeeze merchants and thwart rivals. For consumers, that’s led to artificially inflated prices and a
degraded shopping experience, the agency alleges. In the 172-page suit, the FTC said Amazon relies on an “anti-discounting strategy” and a “massive web-crawling
apparatus that constantly tracks online prices” to stifle competition. The agency said Amazon punishes third-party sellers who offer cheaper products elsewhere
by threatening to disqualify them from appearing in the buy box if it detects a lower price. Losing the buy box is an “existential threat” to sellers’
businesses, the complaint alleges. The end result of these tactics, the FTC argues, is elevated prices across the web.
</item 16>

<item 17>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Google, which is also seeking to capitalise on surging demand for AI technology, saw revenue climb 11 per cent to $US76.7 billion, with net income of $US19.7
billion thanks to a recovery in digital advertising revenue. However, its closely watched cloud division, which houses its data storage business and many of its
AI efforts, missed Wall Street expectations. “Investors were disappointed by the relatively weak performance at its Google cloud platform, which is at risk of
falling further behind” competitors’ offerings, Jesse Cohen, a senior analyst at Investing.com, wrote in an email. Loading During trading hours, the S&P 500
slipped off session highs, led by losses in energy shares as oil slid below $US84 a barrel, but once again the index found support above the key 4200 mark.
Verizon Communications, 3M and General Electric climbed on bullish forecasts. Facebook and Instagram owner Meta dropped after being sued by California and dozen
of other US states over harmful youth marketing claims.
</item 17>

<item 18>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“I’ve lost a lot of sleep because of some of the things that they have done. And the issues that they brought up, while they are interesting, they haven’t
created me a lot of pain.” Needham said he was particularly puzzled by the inclusion of the claims that Amazon is coercive in the way it encourages sellers to
use its fulfillment service, known as Fulfillment by Amazon, or FBA. Needham said many sellers “love FBA” because of its compelling value in terms of the price
and promise to deliver two-day shipping. For many, using FBA doesn’t feel like a requirement, but they believe using it will make their businesses "easier and
more effective." "I think that the power that Amazon wields over sellers is considerable and absolutely worth looking into,” Needham said. “But I’m not sure if
this would actually change that.”
</item 18>

<item 19>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
Rogers’ corporate governance committee is conducting an investigation of the events that took place in 2021, documents show. The committee has asked for copies
of “all correspondence” between Rogers-Hixon and former CEO Natale, three other former executives, the five ousted directors, Martha Rogers, journalists and
others, according to a letter to Rogers-Hixon that’s included with the court filing.

“The investigation is intended to inform decisions that may be made about your status as a director of RCI,” says the letter signed by Robert Gemmell, a former
Citigroup Inc. banker who is Rogers’ lead director.

Although the company is controlled by a family trust, it’s structured in a way that gives significant power to the trust’s chair, in this case Edward Rogers,
the only son of late founder Ted Rogers. That’s what allowed him to reshape the board in 2021 over the opposition of family members.

The family dynamics have grown even more complicated in recent months because of the deaths of three key insiders.
</item 19>

<item 20>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Shopify, for example, gives sellers much more control over how they communicate with customers, Needham said, adding that although Amazon still controls the
communication process, at least there is one. "I wish it was a clear-cut case,” Needham said. “I have a vested interest in the marketplace doing really well, as
a seller and as a service provider. And... this case, it doesn’t make the marketplace better for sellers."

Concerns over Amazon pricing policies, fees

Many sellers have zeroed in on Amazon’s pricing policies and rising fees as rightful areas of concern in the FTC's lawsuit. Molson Hart, whose company Viahart
sells toys on Amazon, has been a longtime critic of Amazon’s pricing policies. Hart complained of how Amazon’s seller fees impact pricing in a 2019 Medium post
and later that year testified about his experience before a House committee.
</item 20>
=== END ITEM id=3fa6d24c718ac0ec ===
