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

=== ITEM id=dce612309a9b5bcd ===
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
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
And as they rallied so spectacularly to salvage a match that looked cooked after 35 minutes, running on fumes and adrenaline, sustained by the sizzle of the
Wembley crowd, the cold air on their skin and the flickering candle of qualification for Paris next summer, it was as if a kind of energy was surging within
them.

An energy that we have not really glimpsed from them since they reassembled this autumn, barely a month on from the trauma of Sydney, still a little dazed and
concussed.

Beth Mead came on for Chloe Kelly at half-time, Georgia Stanway moved further forward and made herself a menace in the penalty area, and the old buddies Toone
and Alessia Russo teamed up off the bench. The Dutch retreated further and further into Wembley’s wide open spaces, giving Lauren James the time and territory
to dictate.
</item 1>

<item 2>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
Hetmyer danced out of his crease and got a good piece of the length ball but it flew flat and hard to Duckett on the midwicket sponge. And he does it on his
100th T20I appearance! 💯#BBCCricket #WIvENG pic.twitter.com/LYhUBtNnUj — Test Match Special (@bbctms) December 13, 2023

Updated at 01.02 GMT

13 Dec 2023 00.49 GMT WICKET! Pooran c & b Rehan Ahmed 13 (West Indies 100-3) Rehan back into the attack after his first over was minced for n n n n nineteen.
Got him! Pooran flicks back a full ball and Rehan takes a nifty catch tumbling away to his left. Shimron Hetymer is the new batter and he tucks a single into
the leg side to open his account.
</item 2>

<item 3>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 23.00 GMT 17th over: West Indies 114-5 (Hope 30, Russell 1) The wicket England really need is that of Shai Hope, who is quietly guiding West Indies
towards the finish line.
</item 3>

<item 4>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
Imagine what Manchester United could do if Erik ten Hag hadn’t already lost the dressing room. Yes, it’s been another of those weeks at Old Trafford: rancour,
rumour and recrimination, barbs in the press and barbs at the press, defeat at Newcastle followed by this stirring recovery against Chelsea, and the startling
realisation that United are now three points behind Manchester City. It’s still only Thursday, by the way.

Perhaps ultimately this win only buys Ten Hag a few days’ grace, a warm flume of goodwill that lasts only as long as it takes for Dominic Solanke to run through
on goal for Bournemouth on Saturday afternoon. Back in the grip of crisis, back in the now-familiar lexicon of surly unsourced stories about dressing room
discontent. Such is the way of things at football’s most reliable content provider. This beast must always eat.

But block out the noise for just a moment, as of course United cannot, and something interesting is happening here.
</item 4>

<item 5>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
“Bruno is an [effective] player when we have a lot of the ball. Then he is at his best. And there are games against top teams when you have less of the ball. So
then it is about taking benefit from [any attacking] moments and for the rest you have to work hard for the team. And to get a result because that is the most
important, that the team is winning.”

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply.
</item 5>

<item 6>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
Mitch Marsh, Australia

In the absence of Travis Head, Marsh has played the role of opener with much responsibility for the Aussies, providing good starts as he accumulated 441 runs in
10 matches.

9. Rassie van der Dussen, South Africa

Rassie has been a support system for the Proteas in the middle order. Though sometimes inconsistent, he has accumulated 448 runs in 10 innings. He has two
centuries and two fifties under his belt in his campaign, with a high score of 133 runs.

8. KL Rahul, India

The Indian number five has been on great form scoring 452 runs giving great support to the Indian middle order. He has one century and two fifties to his name
in the World Cup.

7.
</item 6>

<item 7>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
A wicket once so true turned untrustworthy and they struggled to a score of 132, less than half the 267 to which they had joyously motored just 48 hours
previously. And though the home side’s response was anything but overwhelming, neither was the result ever in significant doubt.

“We’ve had some high-scoring games, but the last game was almost the complete opposite of what happened here,” Reece Topley said. “We were probably just under
par with the first-innings score but with the ball we were excellent, pushed them right to the end. I think it’s been an amazing series. Both teams have played
some unbelievable cricket. I was so excited turning up today, because it was basically a final and those are the games you want to play in and be on the right
side of. The bottom line is you want to win this series, I want to win every series I can for England.”
</item 7>

<item 8>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Adil Rashid (right) celebrates the dismissal of Johnson Charles (left). Photograph: Randy Brooks/AFP/Getty Images

Updated at 22.43 GMT

6d ago 22.17 GMT 7th over: West Indies 51-2 (Charles 26, Hope 7) Rehan Ahmed’s first over disappears for 12. It started quite gently, with four singles and a
two off the first five balls, but then Charles clattered a no-nonsense straight six. “We’re not going to win this, Rob,” writes Guy Hornsby. “But I’m just here
to say I’m still watching for Woakes’ seam position and Rehan’s googly. That is all.”
</item 8>

<item 9>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
We’ll be back to cover the second match in Grenada on Thursday, do join us for that. Until then, thank you and goodnight.

Updated at 02.36 GMT

13 Dec 2023 02.28 GMT Andre Russell is Player of the Match: In his first T20I in over two years he took 3-29 and smashed 29* off just 14 deliveries with the
game in the balance. He seems very pleased to be back in maroon. Life is so funny. Since when I got selected for the West Indies two weeks ago I’ve been
dreaming of being man of the match.”
</item 9>

<item 10>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.16 GMT 4th over: West Indies 47-1 (Mayers 18, Hope 5) Tymal Mills comes onto bowl for England for the first time in eighteen months, he has
plenty of T20 skills up his sleeve, can he showcase them here? Hmmm, a couple of wides slanted down the leg side isn’t the best of starts. Mills lets out a
hearty groan at the second. Two full balls are worked for singles… Kablammo! Mills to Mayers – short ball – pulled onto the roof for SIX! What a shot, new ball
please! DROP! And it is Ben Duckett who shells it! It would have been even better than the first grab but the ball smashes into his palm as he leaps to his
right and comes straight out. They stick or they don’t. It’s all happening.
</item 10>

<item 11>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
7. Marco Jansen, South Africa

Standing at a staggering 6 feet 7 inches tall, Jansen has made full use of his height, and it was evident in his bowling too. The 23-year-old picked up 17
wickets in nine matches for the Proteas, conceding 450 runs.

6. Shaheen Shah Afridi, Pakistan

Afridi is the best bowler for Pakistan in this tournament, with 18 wickets and conceding 481 runs in nine matches.

5. Gerald Coetzee, South Africa

The 23-year-old Coetzee has picked up 20 wickets for the Proteas in eight matches, conceding 396 runs.

4.
</item 11>

<item 12>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
Sarina Wiegman talks a lot about defending against counterattacks, because this is basically the key to the whole enterprise. Cut off the counter, and with
England’s technical quality they can easily dominate. It is also the essential measure of how the team is functioning as a collective: the efficiency of the
press, the closing of the angles, the command of space. Lineth Beerensteyn’s two goals were the sort England simply were not conceding four months ago.

Mary Earps shows her frustration after conceding a second goal to the Netherlands. Photograph: Tom Jenkins/The Guardian

So England passed the ball around a bit and gave it away a bit and essentially ran around trying to feel normal again, and on countless occasions Kelly or Hemp
would win a first or second ball, look up and see a penalty area empty of white shirts.
</item 12>

<item 13>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
These are issues not so much of tactics but cohesion and energy: the timing of runs, the sharpness required not just to anticipate where the ball is going to
be, but actually get there. You can still chase when you’re running on fumes, but to create: this needs an idea and a vision, the clarity of mind to see the
picture before it has materialised.

skip past newsletter promotion Sign up to Moving the Goalposts Free weekly newsletter No topic is too small or too big for us to cover as we deliver a weekly
roundup of the wonderful world of women’s football Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside
parties. For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply.
</item 13>

<item 14>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Updated at 23.20 GMT

6d ago 23.16 GMT I suppose it’s appropriate that it ended with a six, the 120th of the series. Hope played a gem of an innings there, calmly anchoring West
Indies in an increasingly nervous chase with a run-a-ball 43. England fought admirably to defend a below-par total; but for Hope, they would probably have done
it.

Updated at 23.17 GMT

6d ago 23.14 GMT WEST INDIES WIN THE SERIES 3-2! 19.2 overs: West Indies 133-6 (Hope 43, Holder 4) That’s it! Hope pings Woakes over extra cover for six to seal
another series victory for West Indies! West Indies’ Shai Hope (right) celebrates with Jason Holder after hitting a six to win the match by 4 wickets.
</item 14>

<item 15>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.33 GMT 11th over: West Indies 71-3 (Hope 14, Rutherford 12) After 23 balls without a West Indian boundary, Rutherford lifts Moeen handsomely over
extra cover for four. That’s an excellent shot which keeps West Indies in control of the run-chase.

Updated at 22.33 GMT

6d ago 22.28 GMT 10th over: West Indies 62-3 (Hope 12, Rutherford 5) West Indies know how dangerous Rashid is and are content to sit on him. I think that was
Charles’ plan as well, but then he saw a low full toss and his brain went rogue. Time for drinks, after which West Indies need 71 from 60 balls.



6d ago 22.25 GMT Rutherford is not out! I need someone else’s glasses: it was comfortably missing leg stump and England have lost a review.
</item 15>

<item 16>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.45 GMT 14th over: West Indies 94-3 (Hope 19, Rutherford 30) West Indies are rushing towards a series victory. Rutherford hooks the new bowler Sam
Curran for six more, and there are 11 from the over in total. Rutherford has 30 from 23 balls, Hope 19 from 28. They’re playing their roles to perfection.



6d ago 22.40 GMT 13th over: West Indies 83-3 (Hope 17, Rutherford 21) Rehan Ahmed returns, perhaps in the hope of tempting Rutherford into one big shot too
many. Instead it’s Shai Hope, who is happily playing the anchor role, who faces five of the six deliveries. England are running out of time.
</item 16>

<item 17>
Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see (The Roar | Sports Writers Blog,
2023-12-07T22:28:31+00:00)
“We’ll see what the appeal brings. The mentality is key for me, we took a knock but there have been a lot of knocks here over the last couple of years.

Advertisement

“But the team mentality is growing. I can’t emphasise it enough, the commitment to each other and the connection is a powerful thing because I know we’ve got
quality.”

Eddie Howe’s side are proving one of the most inconsistent in the league, albeit with a lengthy injury list.

In the last month, they have defeated Manchester United, Arsenal and Chelsea – as well as drawing at French heavyweights PSG in the Champions League – but have
now lost comfortably to two cellar dwellers in Everton and Bournemouth.

“We have to do better,” said Howe.

“There are reasons behind every performance and we have struggled to change players. That is a big miss with the quality of players out. The longer you do it,
the harder it gets.
</item 17>

<item 18>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
In the episode, ‘ecky thump’ is a secret Lancastrian martial art using black puddings as a weapon. I’m not suggesting we go as far introducing black puddings
immediately. A phased withdrawal of the cricket bat will cause less fuss.” But what would you call The Hundred? Bloody Eck?

Updated at 22.00 GMT

6d ago 21.54 GMT WICKET! West Indies 20-1 (King c Buttler b Topley 3) One down, nine to go. Brandon King slogs Topley miles in the air towards mid-on, where
Buttler takes a simple catch. Jos Buttler takes the catch to dismiss Brandon King, and the first West Indies wicket is gone. Photograph: Randy Brooks/AFP/Getty
Images King traipses off the pitch.
</item 18>

<item 19>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

“I think that it is important sometimes that you have to go through those moments to realise what it takes. There are elements that happen in a football match
and elements that happen in the buildup to a football match, so you have to experience all those things if you want to be successful.”

The win against Everton on Sunday was defined by two teenagers, Alejandro Garnacho and Kobbie Mainoo.
</item 19>

<item 20>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
“Morgan’s England team was a finely tuned machine with everyone knowing their role and most of them at the top of their form… all Morgan had to do was pull the
strings of a great unit of cricketers and he did it well, but my Mum could have captained that lot.

“On the other hand Jos has a dysfunctional squad of players. Poorly selected, poorly prepared, not a settled team, many not sure of their roles, patchy form,
confidence wobbly and to cap it all England’s best two and most influential white-ball players have not been available.”

Boycott’s voice was added to by former England Test captains Nasser Hussain and Michael Atherton, who have both decried the sudden decline of the team’s
previously dominant white-ball set-up, having entered the World Cup as reigning ODI and T20I champions.
</item 20>

<item 21>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
Three looming ODIs against India in the next week and three ensuing T20s will be the first showing of that, with the hosts also a rising force in the white-ball
game.

Advertisement

“Within our group we’re really comfortable with where we’re at, and the progression we’re on,” Healy said.

“Ultimately there’s a big T20 World Cup in Bangladesh towards the end of next year, that is a real carrot for us.

“These are the sort of big trophies that we’re wanting to win.

“And yes, every series in between now and then is important, but at the same time we’re building towards that World Cup.

“Maybe you can ask me that (supremacy) question at the end of the six white-ball games to see how we’re going and we can answer that.

“These next six games will probably give us a better indication about how we’re tracking but also what we might need to improve on.”
</item 21>

<item 22>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
Previously he has never got more than seven. Where is this coming from? Partly from the fact that opponents have still not quite cottoned on to McTominay’s goal
threat, those late runs to the far post, the blur on the edge of the 18-yard box that you don’t spot until it’s too late. But partly, too, it is a function of
how Ten Hag is using him.

Erik ten Hag (right) and Mitchell van der Gaag, his assistant, talk tactics during the game. Photograph: Ash Donelon/Manchester United/Getty Images

Unlike his previous more conventional midfield roles, these days McTominay basically has a licence to create overloads wherever he finds them. Against low
blocks and structured defences – and nobody is remotely accusing Chelsea of possessing one of those – he provides the extra body in a tightly marked penalty
area.
</item 22>

<item 23>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
12.4 overs: West Indies 108-4 (Hope 27, Powell 6)

Updated at 01.09 GMT

13 Dec 2023 00.58 GMT 12th over: West Indies 102-4 (Hope 26, Hetmyer 1) Just a single and the wicket off the over, Shai Hope can’t break the legspin shackles.
Windies skipper Rovman Powell is the new man and Buttler cranks the pressure by inserting a slip.

Updated at 01.04 GMT

13 Dec 2023 00.52 GMT WICKET! Hetmyer c Duckett b Rashid 1 (West Indies 101-4) Adil Rashid takes his 100th T20I wicket! A crucial one in the context of the game
too – the dangerous Hetmyer is sent on his way for just a single as England look to ramp up the pressure heading into the nub end of this match.
</item 23>

<item 24>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
“Some of those cricketers on that park today would go in your best ever England white ball XI. It’s just been a bridge too far for most of them.

“What I don’t like is giving players a cop out… when we win the 50-over World Cup and the 20-over World Cup, aren’t they great? We’re brilliant. And when the
wheels come of it’s the structure. It’s the structure of English cricket, we’re a disgrace.

“We play 20-over cricket, we play 100-ball cricket, we don’t play enough 50-over cricket – how much 50-over cricket domestically has Virat Kohli played? Or
Heinrich Klaasen or anyone out here?

“That’s what’s made this great side over the last six years, travelling round the world playing T20 franchises. It’s such a lame excuse.”
</item 24>

<item 25>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Photograph: Randy Brooks/AFP/Getty Images King traipses off the pitch. Photograph: Randy Brooks/AFP/Getty Images

Updated at 22.14 GMT

6d ago 21.54 GMT 2nd over: West Indies 20-0 (King 3, Charles 13) Johnson Charles was playher of the match when West Indies beat England at the T20 World Cup 11
years ago. Charles walloped 84 off 56 balls that day, and he looks like he fancies something similar here. He pulls Woakes for a big six, albeit with a hint of
top-edge, and glides a boundary past short third man. West Indies are off to a flyer.
</item 25>

<item 26>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
Top 10 Bowlers of ODI World Cup 2023:

Rank Player Mat Wkts 1 Mohammed Shami (IND) 7 23 2 A Zampa (AUS) 11 23 3 D Madushanka (SL) 9 21 4 JJ Bumrah (IND) 11 20 5 G Coetzee (SA) 8 20 6 Shaheen Shah
Afridi (PAK) 9 18 7 M Jansen (SA) 9 17 8 RA Jadeja (IND) 11 16 9 Josh Hazlewood 11 16 10 MJ Santner (NZ) 10 16 11 BFW de Leede (NED) 9 16

*Still Bowling

10.
</item 26>

<item 27>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
It would have been four but for an outstanding stop from Rehan Ahmed.



6d ago 23.12 GMT 19th over; West Indies 124-6 (Hope 37, Holder 1) Holder is almost run out first ball! He took a very dodgy single to mid-off and was well short
of the crease when Curran just failed to gather Rashid’s throw. Nine to win from six balls, Chris Woakes to bowl them.



6d ago 23.10 GMT WICKET! West Indies 123-6 (Russell c Livingstone b Curran 3) Oh my. Russell slugs a low full toss straight down the throat of Livingstone at
long-on, and England – who have been behind the game throughout this innings – are again dreaming of an improbable victory. West Indies need 10 from 7 balls.
</item 27>

<item 28>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
But the first real in-match statement came on 27 minutes as New Zealand-born Aki – perhaps the player of the World Cup so far who had made a tournament-high 61
carries, 33 dominant carries and 23 defenders beaten heading into the weekend – brilliantly jinked inside two defenders, fended off another and powered through
a gap to dive over the line.

Yet more All-Black class saw Ardie Savea go over in the corner after quick ball was shipped wide but with scrum-half Aaron Smith in the sin-bin for a deliberate
knock-on, Ireland narrowed the gap to 18-17 just before the break as another New Zealand-born Irish star, Jamison Gibson-Park, sniped from a lineout maul and
somehow stretched out for the try.

The All Blacks extended an incredible record of leading at half-time of their quarter-final in every single World Cup but that lead had never felt more fragile.
</item 28>

<item 29>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
1. Mohammad Shami, India

Shami announced his return to the World Cup with a bang, picking a fifer in the very first match he played for India. He has picked up 24 wickets so far in just
six matches, conceding just 257 runs. He has picked up three five-wicket hauls including a seven-wicket haul against New Zealand in the semi-final on Wednesday,
November 15 at the Wankhede Stadium in Mumbai.

Catch the latest stock market updates here. For all other news related to business, politics, tech, sports, and auto, visit Zeebiz.com.
</item 29>

<item 30>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
“Certain things [here] are different,” shrugs Ten Hag when asked why the No 1 is taking time to settle in England. “OK it is 11-v-11, and it is about attacking,
defending, transition. But there is more: the approach to the game, the speed of the game, the intensity. The varieties in styles. So there is absolutely a
difference and players need to adjust, need time to do so.”

Last season’s derby at Old Trafford ended in a 2-1 win for United. Bruno Fernandes’s equaliser was controversial with Marcus Rashford offside but not so the
goal was chalked off as he had not touched the ball. So far this term the captain has scored two goals in 12 United appearances and Ten Hag has no doubt he can
be a major factor in Sunday’s showdown.

“If he does [score] again, I’ll be happy,” the manager says. “Bruno is an [effective] player when we have a lot of the ball.
</item 30>

<item 31>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
What is the style of play, what’s the plan?

“He needs to take a step back, ten Hag, and say what do I need to do with these players I have available in order to first of all stop leaking goals, because
that’s a problem, and secondly, score goals – how do I get the next goal? Four games without a goal, it’s a long time.”

Advertisement

DEADLOCK BROKEN! West Ham lead against Manchester United! ????



That assist from Lucas Paquetá though… ????



Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/Qqq7zugQmj — Optus Sport (@OptusSport) December 23, 2023

Ten Hag tried to put a positive spin on the performance by claiming to BBC that they were the better side before the first goal.
</item 31>

<item 32>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
James had a hand in all three goals, and as Toone ghosted in at the far post to provide the final flourish, Wembley screeched out one last call to arms, one
last push, one last effort.

For all this, it is probably worth pointing out that for large parts of the game England were not very good. And we were reminded that for all their creativity
and verve going forward perhaps the biggest flaw in this side is a lack of presence in the two penalty areas, the physicality and the alpha energy to do the
grunt work, to convert the crosses and to head them away.

Jess Carter had a poor game and was withdrawn for Esme Morgan after half-time, Alex Greenwood over-committed for the first Dutch goal and even the
all-conquering Mary Earps plummeted to Earth with a poor first-half error.

But beyond personnel, a basic brittleness seems to have crept into this team, a sense of melting surfaces and cracks in the ice.
</item 32>

<item 33>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.05 GMT 4th over: West Indies 34-2 (Charles 15, Hope 1) England needed early wickets to have a realistic chance of victory. One more in the Powerplay,
before they bring on Adil Rashid, would put them right back in the game.



6d ago 22.03 GMT WICKET! West Indies 33-2 (Pooran b Woakes 10) UltraEdge shows Pooran did edge that ball from Topley – but it doesn’t matter because Woakes has
got him! Pooran tried to ride the bounce of an awkward delivery that cramped him for room, took the inside of an angled bat and ricocheted onto the stumps.
Nicholas Pooran looks glum as his bails go flying.
</item 33>

<item 34>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
Brutal elegance. Rehan comes back well with two sharply turning leggies. Bosh! Shai Hope gets in on the act and smears a length ball over mid on for the third
six of the over. It’s a learning curve for the tyro spinner but that doesn’t make it any less bruising.



13 Dec 2023 00.23 GMT 6th over: West Indies 59-1 (Mayers 22, Hope 13) Curran is called back, Mills granted just the one over for the time being. Two slower
balls bring two soothing dots for England. A single each to Mayers and Hope. Curran does very well to get out of the over with just three runs conceded.
Powerplay complete. How long till we see some spin from Rehan? Not long at all, he’s on for the next over.
</item 34>

<item 35>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
Russell at his best adds so much to West Indies's side. With him at 7, and Holder, Shepherd and Hosein to follow, West Indies bat incredibly deep — Tim Wigmore
(@timwig) December 12, 2023

Updated at 23.48 GMT

12 Dec 2023 23.46 GMT England all out 171! Tymal Mills gets a gossamer thin edge through to Pooran and England are all out with three deliveries to spare! What
a turnaround for West Indies who were staring down the barrel with England flying 77-0 after six overs.



12 Dec 2023 23.44 GMT WICKET! Rashid b Joseph 0 (England 170-9) Rashid goes to swipe Joseph in the final over and hacks the ball onto his timbers. England nine
down…
</item 35>

<item 36>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Australia posted a target of 329 runs, losing seven wickets in the process. Steven Smith scored a brilliant 105 runs off 93 balls, with Aaron Finch supporting
the innings with an 81-run knock of 116 balls. Umesh Yadav picked up four wickets for India. India crumbled within 233 runs in 46.5 overs. MS Dhoni was the
highest scorer for the Indian side in that match, with 65 runs off 65 balls.

James Faulkner picked up three wickets for Australia, with Mitchell Johnson and Mitchell Stark taking two wickets each.

Australia went all the way to lift their fifth ODI World Cup trophy, defeating New Zealand in the final.

2019, Match 14, ICC Cricket World Cup, The Oval

India took their revenge in this World Cup as they annihilated the Australian bowlers by posting a target of 353 runs in 50 overs.
</item 36>

<item 37>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
That’s Rashid’s 99th T20I wicket – in truth it was a bit of a drag down but from the front of the hand so it skidded onto Mayers who didn’t time it and was
caught by Brook in the deep. England needed that BIG TIME. A masterclass from Rashid in the rest of the over, Nicholas Pooran is the new man and his clip for
one is the only run from the over. 8th over: West Indies 79-2 (Hope 19, Pooran 1)

Updated at 00.56 GMT

13 Dec 2023 00.28 GMT 7th over: West Indies 78-1 (Mayers 35, Hope 19) Rehan to Kyle Mayers. First ball is dragged down and launched over the midwicket boundary
for SIX. Pressure on the young leggie. What a shot! Mayers holds the pose as he lofts a supreme drive down the ground for SIX more. Brutal elegance.
</item 37>

<item 38>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 23.09 GMT 18.4 overs: West Indies 123-5 (Hope 37, Russell 3) Sam Curran concedes only one run from the first four balls of a superb penultimate over. The
ball is getting very wet, though, so there’s a short break in play while England call for a dry towel. West Indies need 10 from 8 balls.



6d ago 23.04 GMT 18th over: West Indies 122-5 (Hope 36, Russell 3) Jos Buttler gambles by continuing with Rehan Ahmed, who has been relatively expensive. It
doesn’t come off: Hope skims a back cut between short third and backward point for four to move the requirement back below a run a ball. It’s very hard to see
how England can with this now.
</item 38>

<item 39>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/IcKGdmJv4O — Optus Sport (@OptusSport) December 23, 2023

Manchester United legend Peter Schmeichel lashed the team’s lack of a plan and blasted them for looking slow against the Hammers.

“It was just in many ways irrelevant. There was no plan to it,” he said. “West Ham got an easy win which is a little bit silly to say.

“When I look at the team, I don’t know what the identity of the team is. What is the style of play, what’s the plan?
</item 39>

<item 40>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
12 Dec 2023 23.42 GMT WICKET! Ahmed c Powell b Russell 1 (England 169-8) Rehan Ahmed follows shortly after, trying to smear a length ball down the ground he can
only find Powell lurking ten yards in from the sponge. Russell gets two in the over and finishes his spell and his first T20I in two years with figures of 3-19.
Nicely done. 19th over: England 169-8 (Woakes 2, Rashid 0)



12 Dec 2023 23.37 GMT WICKET! Livingstone b Russell 27 (England 167-7) Livingstone chops a cutter from Russell onto his stumps!
</item 40>

<item 41>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
Hussain said post-match on Sky Sports that “back to back, I’ve not seen England play as badly as that”, adding that the team’s issues are less to do with the
frequently-criticised The Hundred or a lack of 50-over cricket in the build-up to the tournament as it is the sudden and collective loss of form of the English
batting order.

“If you walk in that dressing room now and asked them to put their hand on their heart and say, are you in good nick? I reckon maybe one or two could say that,”
Hussain said, later doubling down with a scathing editorial in the Daily Mail.

Advertisement

“They have all collapsed as a unit and their form has deserted them. Compare that to South Africa and their batting lineup, everyone’s in nick. You look at
India and their batting lineup, Rohit [Sharma], KL Rahul, Virat Kohli, Shubman Gill – all in nick.
</item 41>

<item 42>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.25 GMT ENGLAND REVIEW FOR LBW AGAINST RUTHERFORD! This looks really close. Rutherford pushed around a legbreak from Rashid and was hit on the pad in
front of middle and leg.



6d ago 22.24 GMT 9th over: West Indies 58-3 (Hope 11, Rutherford 2) Moeen replaces Rehan and rushes through a pretty good over, four singles from it. West
Indies are still on top but another wicket would make things very interesting.



6d ago 22.21 GMT WICKET! West Indies 54-3 (Charles c Buttler b Rashid 27) Adil Rashid strikes straight away. On this occasion it had nothing to do with his
bowling, but maybe a little to do with his aura. He accidentally bowled a very wide, dipping full toss to Charles, who clunked it low to Jos Buttler at short
extra cover.
</item 42>

<item 43>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
England’s hopes of defending their World Cup crown are in tatters – and we’re barely halfway through the 2023 tournament.

Second-last of ten on the table after a fourth abject loss in barely three weeks, this time by eight wickets and nearly 25 overs to spare against Sri Lanka
having been bowled out for a miserable 156 in Bengaluru, the 2019 champions are now only mathematically a chance of scraping through to the semi-finals.

They need not only to somehow turn their horrendous form around to win all four of their remaining matches, including an upcoming date with table-toppers and
tournament favourites India, but also to hope numerous other results, including Australia to lose at least two and probably three of its own four games, to
reach the knockout stage.

Facebook Twitter Whatsapp Reddit Email Share

As expected, the reaction from the UK, and from a swathe of English cricket greats, was scathing.
</item 43>

<item 44>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.37 GMT 9th over: West Indies 87-2 (Hope 23, Pooran 3) Will the wickets slow the Windies as they did England? Rehan is replaced by Tymal Mills who
starts with another wide. Rusty stuff so far from the southpaw seamer. That’s better – a series of slower balls and back of the hand trickery restricts Pooran
and Hope before the final ball is top edged over Buttler’s head for a one bounce four. West Indies need 85 from 66 balls.



13 Dec 2023 00.32 GMT WICKET! Kyle Mayers c Brook b Rashid 35 (West Indies 79-3) The sorcerer follows his apprentice and bags a wicket first ball!
</item 44>

<item 45>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 21.31 GMT 19th over: England 129-8 (Curran 9, Rashid 1) The hat-trick ball is another attempted yorker that Rashid flicks wide of Russell for a single.
He would have had three wickets in four balls had Hosein taken a tricky low chance at third man to dismiss Curran. Russell, who bowled really well, finishes
with two for 25.
</item 45>

<item 46>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 21.39 GMT The difference two days make Tuesday, Tarouba England 267-3, 19 sixes

Thursday, Taroura England 132 all out, five sixes

Updated at 21.45 GMT

6d ago 21.36 GMT WICKET! England 132 all out (Curran c Rutherford b Holder 12) Sam Curran clouts the next ball to long-off to end an increasingly miserable
England innings. They were 109 for four after 14 overs, and then the house fell down. West Indies need 133 to win the series.

Updated at 21.38 GMT

6d ago 21.33 GMT WICKET! England 132-9 (Rashid run out 1) Adil Rashid sacrifices himself, just like in the 2019 World Cup final, so that Curran can stay on
strike. Four balls remaining.
</item 46>

<item 47>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Ian Foster, who made two changes to his starting side and three in total, was forced to defend his centre’s actions while twisting the knife into Sexton’s
finish.

“It happens on most sporting parks when tensions are high and a lot at stake,” Foster said.

“Unfortunately recently it’s also included players saying a few things to referees. Is it right for the game? I don’t know. You don’t hear a lot of players
complaining about it.”

Advertisement

Captain Sam Cane added: “It’s players’ different personalities and emotions running high in those moments. There’s always people who have different tactics but
it’s always just left out there.”

Foster recalled Sam Whitelock to the starting side in the only change to the pack.

Whitelock, whose breakdown penalty win after 37 phases denied Ireland a match-winning penalty, has been recalled and will partner Scott Barrett in the
second-row. Brodie Retallick will come off the bench.
</item 47>

<item 48>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Amid the wider theme of transformation Adil Rashid’s excellence remains constant. The 35-year-old, who rose to the top of the T20 bowling rankings for the first
time on Wednesday, took two wickets and conceded just 21 while Topley was even better, also claiming a pair of victims but conceding only 17. England take
plenty of positives from this series, but the result was not one.
</item 48>

<item 49>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Nicholas Pooran looks glum as his bails go flying. Photograph: Ricardo Mazalán/AP

Updated at 22.15 GMT

6d ago 21.59 GMT 3rd over: West Indies 30-1 (Charles 13, Pooran 9) Nicholas Pooran strolls to the crease and launches his second ball over midwicket for six.
And why not. England thought he had edged the next ball to the keeper, though eventually they ran out of time to review. West Indies need 103 from 17 overs.
“Evening Rob,” says Kim Thonger. “Jolly exciting this game, in a damaged sort of way. It occurs to me perhaps T20 internationals need a complete rebrand. May I
propose we rename them Ecky Thump. The term was Introduced in Kung Fu Kapers, a 1975 episode of BBC comedy The Goodies.
</item 49>

<item 50>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
Alyssa Healy has rejected any suggestion Australia’s stranglehold on international women’s cricket is slipping, saying their Test loss to India is barely a blip
on the team’s record.

Australia stumbled to their first Test defeat to India by eight wickets in Mumbai on Sunday, after being outplayed by the hosts in foreign spinning conditions.

The loss comes after one of the more difficult periods for the Australian side in the past six years, with the retirements of Meg Lanning and Rachael Haynes.

Facebook Twitter Whatsapp Reddit Email Share

After winning their fourth straight global ICC event earlier this year in the 50-over World Cup, Australia retained the Ashes via a drawn multi-format series in
England in July.

That tour included losing four of six white-ball matches, with Australia’s win in the Test enough to have them level on points with England and hold onto the
trophy.
</item 50>

<item 51>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India seamer Javagal Srinath took three top-order wickets, including Sanath Jayasuriya and Romesh Kaluwitharana.

Fans at the Eden Garden were cheering loudly till Tendulkar was at the crease.

But once he fell for 65 with the team at 98, India suffered one of the worst collapses in the batting history of the World Cups.

The crowd turned angry and hostile towards Sri Lanka, creating a ruckus at the stadium, and the match was interrupted.

With no respite from onlookers' side, umpires decided to call off the match when India were reeling at 120 for eight in 34.1 overs.

The long-lasting memories of the crowd's bad behaviour and a weeping Vinod Kambli returning to the pavilion are still fresh in the memories of the cricket fans
of that generation.
</item 51>

<item 52>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Photograph: Ricardo Mazalán/AP 𝗔 𝗦𝗜𝗫 𝗧𝗢 𝗙𝗜𝗡𝗜𝗦𝗛! 🙌@WindiesCricket claim the series 3-2 and a white-ball clean-sweep 🏏#WIvENG pic.twitter.com/chrYkHbtcp — Cricket
on TNT Sports (@cricketontnt) December 21, 2023

Updated at 23.34 GMT

6d ago 23.13 GMT 19.1 overs: West Indies 127-6 (Hope 37, Holder 4) Holder inside edges Woakes this far wide of leg stump for three runs. It would have been four
but for an outstanding stop from Rehan Ahmed.
</item 52>

<item 53>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
It’s such a lame excuse.”

Atherton described the England players as looking ‘dazed and shell-shocked, unable to comprehend what has befallen them’, and wrote in The Times that the Sri
Lankan defeat officially represented the ‘end of an era’ for English white-ball supremacy.

Advertisement

“Each defeat, now four in five matches, has chipped away at the confidence of a group of players at the end of a cycle,” Atherton wrote.

“No batsman looks in any kind of form. Every player has been used; some discarded and used again. They lost faith in their all-rounder strategy before the South
Africa match, making three changes, and then went back to it here, making three more.

“Players sense uncertainty in decision-making – and there has been a lot of uncertainty.

“It was hard to think that they could put in a worse performance than in Mumbai but they managed it with the bat in Bengaluru.”
</item 53>

<item 54>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
4. Jasprit Bumrah, India

Bumrah has taken 20 wickets under his belt, conceding 373 runs. He also has the best economy rate in the whole tournament, with 4.06.

3. Dilshan Madushanka, Sri Lanka

Despite such a dreadful performance by the Sri Lankan side in this World Cup, Madhushanka managed to be in the top 2 on the highest wicket-takers list. He
picked up 21 wickets, conceding 525 runs.

2. Adam Zampa, Australia

Adam Zampa is having a World Cup to remember, as he is the highest wicket-taker for the Aussies. He picked up 23 wickets, in which there were three four-wicket
hauls, conceding 515 runs.

1.
</item 54>

<item 55>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
West Indies are off to a flyer.



6d ago 21.49 GMT 1st over: West Indies 7-0 (King 1, Charles 2) There’s some early inswing to the right-handers for Reece Topley. Brandon King survives an LBW
appeal first ball – it was missing leg – before another inswinger to Johnson Charles runs away for four leg-byes. Every boundary hurts when the target is so
low, even in the first over.



6d ago 21.45 GMT After a very short break, the players are back on the field.
</item 55>

<item 56>
Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see (The Roar | Sports Writers Blog,
2023-12-07T22:28:31+00:00)
The longer you do it, the harder it gets. I am disappointed, a missed opportunity, we did not grab it.

“The game was there for us to win in the second half. The first half was even and we had them penned in in the second half. We conceded the first goal and it
changed the momentum of the match. Frustrating and a chance missed.”
</item 56>

<item 57>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 02.36 GMT James Wallace That’s it from me this evening/morning. Time to haul myself out of the OBO armchair and skulk off for some kip. That was an
intriguing game and sets up this T20I series nicely. A real statement has been laid down by Rovman Powell and his men, West Indies were up against it early
doors with England crunching their way to 112-2 off the first ten overs but Jos Buttler’s side only managed to make 59 runs off the next ten, losing eight
wickets along the way. A total of 171 was thirty or more shy of where England would have liked to have ended up and so it proved as West Indies peeled up their
sleeves and showed how muscular their own batting card is – the home side cleared the ropes fourteen times compared to England’s six. Andre Russell and Rovman
Powell found the stands with ease at the business end of the match to get West Indies over the line with eleven balls remaining.
</item 57>

<item 58>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 01.45 GMT WICKET! Shai Hope c Brook b Rehan Ahmed 36 (West Indies 123-5) Ahmed is bunted for SIX by Shai Hope but strikes with the next ball as Hope
holes out!



13 Dec 2023 01.44 GMT 14th over: West Indies 112-4 (Hope 27, Powell 10) Liam Livingstone instead of Rashid and he does well to keep it to singles, England
trying to take this as deep as possible and hope West Indies fluff their lines at the death. West Indies need 55 from 36 balls.
</item 58>

<item 59>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
Leading the charge was the always outspoken Sir Geoff Boycott, who wrote in The Telegraph that the team were paying the price for poor preparation in the
lead-up to the World Cup, as well as shoddy team selection extending back to The Ashes earlier this year.

“It is an old saying: “By failing to prepare, you prepare to fail”,” Boycott wrote.

“History shows that England rarely bat well in India where the ball spins. Our guys are okay on flat pitches but if it turns they have a problem.

“So the best way to try and overcome that would have been for our squad of players to get to India early and play four or five warm-up matches. What did England
do?

Advertisement

Plan only two warm-up games with one getting rained off. Not smart planning… batting and bowling against stiff opponents sharpens your game and puts you on your
mettle. Against Sri Lanka, they were once again exposed.
</item 59>

<item 60>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 02.09 GMT West Indies win by 4 wickets! Andre Russell smashes the winning runs off Sam Curran! A length ball is spanked over point for a one bounce
four to give the home side the win, they go 1-0 up in the series.

Updated at 02.29 GMT

13 Dec 2023 02.07 GMT 18th over: West Indies 168-6 (Powell 31, Russell 25) Adil Rashid is recalled for his final over, can he turn the game back to England?
Nope. ANDRE RUSSELL SMASHES THE RASHID GOOGLY CLEAN OUT OF THE GROUND! Huge SIX! The big man even allows himself a coy smile. He enjoyed that one. A dot the
next ball is followed by a powerful drive all along the baize for four! Eleven in total off the over and West Indies now just need four runs.
</item 60>

<item 61>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
after newsletter promotion

And clarity is perhaps what this team has been lacking most of late: clarity of selection, clarity of formation, clarity of mission and motivation. How much
prestige should we really attach to the Nations League, anyway? Is it primarily an Olympic qualification shootout or something worth winning in its own right?
England ended up trying to answer these questions on the hoof, and somehow it was only when their options had finally clarified down to one – survival or
oblivion – that they recovered their poise and verve.

This entire post-pandemic international cycle – a Euros, a World Cup, an Olympics and another Euros squeezed into three years – feels like a kind of sadism
dressed up as a panorama of opportunity. There is a school of thought out there that a fallow summer might not have been the worst thing for this team: time to
reflect and process, time to breathe, time to refresh.

Instead, they now travel to Glasgow on Tuesday for one last push. The minds are still frazzled.
</item 61>

<item 62>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
This was a win that felt a lot tighter than it really was: United inexplicably hanging on at the end despite enjoying 28 shots and an xG tally of 4.07. For long
periods United played some of their most authentically thrilling football of the season, a doctrine of relentless attack that had Diogo Dalot flying forward
from right-back even as the minutes ticked down.

It was also a game that felt more chaotic than it really was. In fact the slippery, slingy nature of the play was largely by design: United hollowing out their
midfield in order to throw bodies forward, Chelsea hollowing out theirs in order to leave men up for the breakaway. It was a game that hung on United’s
determination to keep allowing Chelsea counterattacks, and Chelsea’s determination to keep messing them up. It was a game, above all, that hung on the curious
main character energy of Scott McTominay.
</item 62>

<item 63>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 02.01 GMT 17th over: West Indies 157-6 (Powell 31, Russell 14) Tymal Mills is entrusted. Singles off the first two balls. Powell swipes a full ball
into the deep to collect a couple. SIX! Mills goes short and Powell gets a mighty top edge for a maximum! Fortune favours the brave, the Windies skipper gave
that everything and it could have gone anywhere. Mills sends one down the leg side – wide ball. Pressure on the bowler here… Shot! Short and wide from Mills and
sent to the fence with Swiss clock timing by Powell. Sixteen off the over, looking terminal for Buttler and co. 15 needed from 18 balls.
</item 63>

<item 64>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 01.14 GMT Promising news – the rain seems to be tailing off and a rope is being dragged around the outfield to take off any excess water. I’ll bring
news as soon as it wafts over from Barbados to my sofa.



13 Dec 2023 01.00 GMT Rain stops play (West Indies 108-4 and require 64 runs off 44 balls) What’s all this then? Out of nowhere the heavens crack like an inky
brûlée and rain starts to pour. The players scurry off as the covers are hauled on. Crucially – West Indies are six runs ahead of the Duckworth Lewis Stern
rate. IF the rain doesn’t abate – and it looks heavy at the moment – then the home side will take the spoils.
</item 64>

<item 65>
Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see (The Roar | Sports Writers Blog,
2023-12-07T22:28:31+00:00)
Spursy. pic.twitter.com/DW6mgUVXnW — OptaJoe (@OptaJoe) December 7, 2023

West Ham aren’t far behind, with this result lifting them to ninth, just three points back from Spurs.

“Huge performance because of the quality of the opposition,” said coach David Moyes.

“We were up against it tonight. We had to dig in. Thankfully we just about scraped it.

“The quality Tottenham showed in the first half, I don’t know if we touched the ball for eight or nine minutes.

Advertisement

“We gained a bit of confidence once we got some passes. We wanted to get after them a bit more but by the same breath we were cautious about them picking us
off.

“We are a team who can give some of the good teams a bloody nose, we have that in us.”

Dwight McNeil fires Everton into the lead against Newcastle!
</item 65>

<item 66>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.37 GMT 12th over: West Indies 80-3 (Hope 15, Rutherford 20) England appeal unsuccessfully for caught behind when Rutherford misses a swipe at Rashid.
There’s only one review left and they decide not to risk it. A good job too; there was nothing on UltraEdge. Out of nothing Rutherford launches a full ball from
Rashid down the ground for six, almost knocking himself off his feet in the process. West Indies are managing an awkward chase pretty well.
</item 66>

<item 67>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
12 Dec 2023 23.48 GMT An innings of two halves for England From 77 for no loss to 171 all out with three balls to spare. England's recent white ball approach is
a variable one yielding mixed...often frustrating results #WIvENG — Aatif Nawaz (@AatifNawaz) December 12, 2023



12 Dec 2023 23.47 GMT Time to stick the kettle on and raid the fridge before West Indies start their chase. 3-19 for Andre Russell on his comeback for West
Indies, balancing the attack perfectly.
</item 67>

<item 68>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“In the game in 2020 there were things you could pinpoint. You can’t say that one game means we can do it as there are a lot of games where we haven’t.

“That moment has to be just a part of it. What we have done since then, all the success and failures [matters in our preparation]. The other team are heavy
favourites, everyone is expecting a New Zealand and South Africa final. We just have to focus on what is required.”

Cheika has maintained that his Los Pumas side was too tight earlier in the tournament but has begun to free themselves up since qualifying for the knockout
stages – and rewarding their fans by doing so.

On inspiring the country?

“It’s not easy,” he said.

“We want to inspire not just by our results but by our ambition, by our ability to get back up and overcome obstacles.
</item 68>

<item 69>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“To inspire is not about results, it’s not just one player who played well; no, it’s about our behaviour in general, our ability to be ready to come back from
difficult times and seizing our opportunities.

“There are not a lot of opportunities in life – us playing in a semi-final is even rarer and we want to have no regrets.”
</item 69>

<item 70>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
That shortage of runs has carried over to his patchy one day performances.

“Chris Woakes is an excellent bowler of swing and seam on English type pitches. His record abroad is poor as he does not have the pace or guile for flat batting
surfaces. It is not the lad’s fault. He was selected but how could people picking the squad go against all the statistics.”

Advertisement

However, Boycott isn’t blaming captain Jos Buttler for the team’s woes, saying the skipper has an impossible job pulling together a ‘dysfunctional squad of
players’ and that not even a ‘miracle man’ would have much of a prayer of salvaging any pride from the wreckage in India.

“Too many people are criticising the captaincy of Buttler and judging him against the captaincy of Eoin Morgan when England won the World Cup four years ago,”
he wrote.
</item 70>

<item 71>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
That he can shout down the noise in the short term while building something in the long term. Right now, in the raptured afterglow of a big win, it feels like a
gamble being won. But best to come back and check in a few days, just to make sure.
</item 71>

<item 72>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
He has one century and two fifties to his name in the World Cup.

7. Shreyas Iyer, India

Iyer is the backbone of the Indian middle order; these are the words of Head Coach Rahul Dravid. Iyer has provided good support to the team when needed, scoring
526 runs in 10 matches. He also has three fifties and two centuries to his name.

6. David Warner, Australia

One of the most destructive openers of the modern era, Warner has been providing great starts to Australian batting, scoring 528 runs in ten innings. His
highest score is 163 runs, along with two centuries and two fifties.

5. Daryl Mitchell, New Zealand

Mitchell has been a powerhouse for the Kiwis in the middle order, scoring 520 runs in nine matches.
</item 72>

<item 73>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 01.41 GMT 13th over: West Indies 112-4 (Hope 27, Powell 10) Rehan has two balls left to complete his over. The first ball is flayed for four by
Powell behind point! Valuable runs for West Indies. Rehan’s final ball is a dot, defended by Powell into the off side. West Indies need 60 from 42 balls.

Updated at 01.42 GMT

13 Dec 2023 01.30 GMT Simon Burnton is our man in Barbados and he sends good news! “It’s a 9.35pm restart. We haven’t lost any overs.” We should be underway in
about eight minutes. I believe that 10.10pm local time is the cut off for the match – that’ll be 2.10am here in London town. Excuse me whilst I head to the
kitchen to microdose some Kenco.
</item 73>

<item 74>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
Here is the list of the top 10 batsmen and bowlers in this edition of the ODI World Cups—

Top 10 Batsmen of ODI World Cup 2023:

Rank Player Mat Runs 1 V Kohli (IND) 11 765 2 RG Sharma (IND) 11 597 3 Q de Kock (SA) 10 594 4 R Ravindra (NZ) 10 578 5 DJ Mitchell (NZ) 10 552 6 DA Warner
(AUS) 11 535 7 SS Iyer (IND) 11 530 8 KL Rahul (IND) 11 452 9 HE van der Dussen (SA) 10 448 10 MR Marsh (AUS) 10 441

*Still Batting

10.
</item 74>

<item 75>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
His high score was 130 runs in this campaign, and he provided good support from time to time whenever the team needed it.

4. Rachin Ravindra, New Zealand

The biggest find of this year's World Cup has to be this 23-year-old all-rounder, Rachin Ravindra. Ravindra has smashed three centuries and two fifties in this
campaign and accumulated a total of 578 runs in 10 matches.

3. Quinton de Kock, South Africa

A pivotal part of the South African top-order, de Kock has been a phenomenal batter, scoring 594 runs in 10 matches. de Kock has the most number of centuries in
this tournament, with four so far.

2.
</item 75>

<item 76>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
That made it 70 for four and the game was one ball into its ninth over, at which point Liam Livingstone and Moeen Ali had to focus for a while on disaster
prevention rather than entertainment.

Gudakesh Motie is embraced his his West Indies teammates after dismissing England’s in-form Phil Salt. Photograph: Randy Brooks/AFP/Getty Images

Suddenly bowlers were energised by the prospect of posting, for a change, less-than-humiliating figures. Akeal Hosein’s were outstanding, his four overs going
for just 20 and bringing the wickets of Will Jacks and Moeen, undone by a relay catch on the boundary after combining with Livingstone to add 40 for the fifth
wicket. Livingstone followed precisely two overs later, prodding the ball limply back into the hands of Motie, at which point things really fell apart, the last
five wickets surrendered in 20 balls for just 11 runs.
</item 76>

<item 77>
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
</item 77>

<item 78>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
What a difference two days make. Somehow Tuesday’s ludicrous, record-smashing, high-scoring entertainment was followed by a sequel on the same surface that
eventually delivered thrilling low-scoring entertainment, and a different result claimed in very different style. It was eventually won for West Indies with
four balls to spare and in the most unfitting of ways – with a massive six – by Shai Hope, again proving himself the right man for a run-chase.

So a series marked by stunning power hitting, roaring run rates and an unexpected midway transformation in England’s performances and prospects concluded with a
3-2 triumph for West Indies, who extended their supremacy over England at home to four series in three formats across nearly five years.

England assembled exactly the same players in exactly the same place and exactly the same situation – losing the toss, put in to bat – but this was an entirely
different performance.
</item 78>

<item 79>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
Speaking after the match, Buttler admitted his team had been ‘short of our best by a very long way’, saying there is ‘no clear answer’ as to how to turn their
fortunes around.

“It’s incredibly tough, an incredibly disappointing tournament. As captain, you feel that a lot,” he said.

Advertisement

“We’ve been short of our best by a very long way. I’m disappointed for myself and the boys that we’ve not given a good account of ourselves… if there was one
golden nugget that we’re not doing then we’d pick that up.”

Sports opinion delivered daily

England’s tournament may only stay mathematically alive until Sunday evening (AEST), when they face the daunting challenge of tournament hosts India in Lucknow,
with defeat certain to end their semi-final chances for good.
</item 79>

<item 80>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
ICC Men's ODI World Cup 2023 Most Runs and Wickets: With just the final of the ICC Men's ODI World Cup 2023 won by Australia beating India by six wickets on
Sunday, November 19, at Narendra Modi Stadium in Ahmedabad, the whole world is ready to witness who will be crowned the World Champions in the 50-over format of
the game. India defeated New Zealand by 70 runs in the first semi-final on Wednesday, November 15, at the Wankhede Stadium in Mumbai, and Australia beat South
Africa by three wickets in the second semi-final on Thursday, November 16, at the Eden Gardens in Kolkata.

During all these matches played, we witnessed some great knocks by the batter as well as some brilliant spells by the bowlers, which annihilated the opposition
and kept the audience in awe.
</item 80>

<item 81>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.57 GMT WICKET! West Indies 113-5 (Powell c Rashid b Topley 8) England haven’t given this up. Powell edges Topley to short third man, where Rashid
takes a smart low catch to his left. West Indies need 20 from 21 balls. Reece Topley (right) celebrates the dismissal of Rovman Powell. Photograph: Randy
Brooks/AFP/Getty Images

Updated at 23.07 GMT

6d ago 22.54 GMT 16th over: West Indies 112-4 (Hope 29, Powell 8) Hope, who is playing an exemplary innings, opens the face to steer Rehan for his first
boundary. Rehan has struggled as the series has progressed, an d later in the over Powell muscles him over long-on for six. West Indies are 21 runs away from a
hat-trick of T20 series wins this year: South Africa away, India and England at home.
</item 81>

<item 82>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
Advertisement

Healy called for Australia’s Test series in India to be extended to three matches their loss, their first in 11 Tests against India.

The hosts easily chased down 75 in the fourth innings on Sunday.

All out for 209 in their first innings, Australia looked far more confident in their second effort of 261 despite the wicket showing more signs of wear.

And Healy believes three-Test tours of India would provide a better match-up between the two sides.

“It would create a real contest over three games,” Healy said.

“That would be an unbelievable experience for our group, and probably a true test of of both sides’ abilities.

Advertisement

“The one-off Test, India playing in their home conditions, you’d expect them to be heavy favourites.

“But the way we tried to adapt and continue to learn throughout the four days was really impressive.

“We would love to have another crack at it, show what we learned and put it into practice.”
</item 82>

<item 83>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
In reply, Mohinder Amarnath played a useful innings of 46, while Yashpal Sharma scored a fine 61.

Sandeep Patil played a fiery innings of 51 not out in just 32 balls as India stormed into semifinals for the first time, beating England by six wickets with 5.2
overs to spare.

Also Read: ICC World Cup 2023: List of centuries; Most tons in World Cup so far

1987 Reliance Cup, against England

Four years later, the World Cup was being played outside England for the first time, and it was India which was hosting it.

In a co-incidence, India was against England in the semi-finals once again, at the Wankhede Stadium, Mumbai, on November 5.

This time, England's Man-of-the-Match skipper Graham Gooch smashed 115 runs in 136 balls to lay a solid foundation for his team's fighting total of 254 for 6
wickets.
</item 83>

<item 84>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 01.55 GMT 16th over: West Indies 141-6 (Powell 26, Russell 4) Livingstone continues. Four singles off the first four balls keeps the pressure on the
West Indies. HOLD THAT THOUGHT. Rovman Powell crunches consecutive sixes down the ground with a shrug and swagger to make it sixteen off the over. The scales
tip once more, England need to do away with one or both of Powell and Russell – there’s no way they don’t win this for West Indies if they are there at the end.
31 needed from 24 balls.
</item 84>

<item 85>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.50 GMT 15th over: West Indies 99-4 (Hope 23, Powell 1) Rashid finishes with figures of 4-0-21-2. He’s had a great series: eight wickets at 15 with an
absurd economy rate of 6.26. But he’s going to end on the losing side.



6d ago 22.48 GMT WICKET! West Indies 95-4 (Rutherford c Curran b Rashid 30) Buttler brings back Rashid for his final over, knowing it’s now or never for
England, and gets an immediate reward. Rutherford smashes a drive towards short extra, where Curran dives to his left to take a superb catch. It’s not the
despair, Laura…
</item 85>

<item 86>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India moved to finals and beat Sri Lanka to win their second World Cup title.



2015 World Cup, against Australia

India were playing against Australia in the second semi-final of the 2015 World Cup in Sydney on March 26. Australia outplayed India in almost all departments
of the game.

Australian Man-of-the-Match Steve Smith (105 runs in 93 balls) and Aaron Finch (81 runs in 116 balls) were the guiding forces in taking Australia to 328 runs
for 7 wickets in 50 overs.

In reply, India needed a few big innings, but apart from skipper MS Dhoni (65 runs in 65 balls), Shikhar Dhawan (45 runs in 34 balls), and Ajinkya Rahane (44
runs in 68 balls), the batting order collapsed miserably as India lost the match by 95 runs.



2019 World Cup, against New Zealand

The cricket World Cup returned to England for the fifth time.
</item 86>

<item 87>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.03 GMT 2nd over: West Indies 30-0 (King 21, Mayers 9) Will Jacks to bowl his workmanlike spin at the other end. Uh-Oh! He drops short and Kyle
Mayers murders it, the ball lands on the roof of the stadium. That deserved everything it got. Mayers cuts for three behind point and then King skips jauntily
out of his crease to skim the ball for four down the ground!



12 Dec 2023 23.59 GMT 1st over: West Indies 16-0 (King 16, Mayers) Brandon King gets the home side off to a blistering start, sending Sam Curran down the ground
for four and swiping a brace of SIXES over mid-wicket. Game on!



12 Dec 2023 23.56 GMT No sooner have I sploshed some water on me teabag… the players emerge. West Indies need 172 to win!
</item 87>

<item 88>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Often there is a sense as a tour enters its final days of thoughts turning to home, and perhaps the focus was not as laser-sharp as it might have been for this
game, determination less than absolute to grasp this opportunity to end England’s run of reverses in the Caribbean. Whatever the reason, the pyrotechnics that
rocketed England to victory in the two previous matches never sparked in this one. It took a fabulous delivery from Gudakesh Motie to end Phil Salt’s run of
absurd form with his score a modest 38, but some of the other wickets that fell during the period that defined England’s innings and the game seemed a little
careless.

In the fourth over Jos Buttler casually turned a Jason Holder delivery to fine leg, where Oshane Thomas was positioned, hands cupped. Harry Brook tried to
paddle-sweep Motie but sent the ball only a few feet into the air, giving Nicholas Pooran an easy catch.
</item 88>

<item 89>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.20 GMT 5th over: West Indies 56-1 (Mayers 20, Hope 12) Woakes continues, after five balls he’s gone for just three runs. Can he get out of the
over cheaply? No – A length ball is clubbed down the ground for SIX with a minimum of fuss by Shai Hope. West Indies have come out of the traps as well as
England did, one over of the Powerplay left.
</item 89>

<item 90>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
The wicket has given Ahmed a confidence boost and he gets out of the over with three dots. 11th over: West Indies 101-3 (Hope 25, Hetmyer 1)



13 Dec 2023 00.43 GMT 10th over: West Indies 99-2 (Hope 25, Pooran 13) Rashid is given another after his excellent first over. The googly is spotted by Pooran
and driven down the ground handsomely for SIX! A glide behind point bring a single, Hope returns the favour with a flick into the leg side. A short ball is
panned into the deep by Pooran for a couple to end the over. Bon Jovi O’clock – West Indies need 73 from 60. Time for a quick slurp for the players and a
knuckle stretch for me. Is there anybody alive out there?!
</item 90>

<item 91>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 01.51 GMT 15th over: West Indies 125-6 (Powell 12, Russell 2) Andre Russell the man to face the hattrick ball… Boo! A decent ball on the stumps is
patted back by the big man. Drama in Barbados, Russell works for a couple. Ahmed takes his cap with a broad smile on his face – he hasn’t had it all his own way
but finishes with 3-39 from his set of four. Advantage England? 47 needed from 30 balls.

Updated at 01.54 GMT

13 Dec 2023 01.47 GMT WICKET! Shepherd c Curran b Rehan Ahmed 0 (West Indies 123-6) Rehan Ahmed on a hattrick! Buttler brings Curran into slip and Shepherd
pokes a leggie to him off his very first ball. Great stuff from Rehan who has turned the game England’s way in the space of a minute.
</item 91>

<item 92>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
Against Sri Lanka, they were once again exposed.

“Too many of our guys just want to play crash, bang wallop all the time irrespective of the type of pitch or opposition. Again not smart.”

The brunt of Boycott’s criticism of England’s team composition was directed at the continued struggles of Jonny Bairstow, who has averaged just 25.2 this
tournament, and all-rounder Chris Woakes, whose nightmare campaign continued against Sri Lanka with figures of 0/30 from six overs and a fourth-ball duck.

“Bairstow has struggled since his horrific injury [breaking his leg 12 months ago],” Boycott wrote.

“I have said before I believe Jonny came back too soon and was short of fitness and match practice. In the recent five Ashes Tests his keeping at Lord’s was
poor with missed chances and he made only three significant batting scores out of nine innings.
</item 92>

<item 93>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
Mitch Santner, New Zealand

New Zealand's bowling has mostly depended on Santner, as he has been effective for the Kiwis on the turning tracks of India, as he picked up 16 wickets and
conceded 449 runs in 10 matches.

9. Josh Hazlewood, Australia

The Aussie pacer has been phenomenal with the new ball. He has created movements with the ball which the batters struggle to play. He picked up 16 wickets in 11
matches.

8. Ravindra Jadeja, India

Jadeja has been bowling some beautiful spells in this tournament for India, especially in the middle overs. He picked up 16 wickets, conceding just 398 at an
economy rate of 4.25 in 10 matches.

7.
</item 93>

<item 94>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
Even if you’re not playing well, if the team is not playing well, or you’re not playing to the standards you expect, you still have to win – so find a way to
win, be pragmatic.”

United have been so of late, claiming victories over Sheffield United, Brentford and FC Copenhagen in their last three matches. But Ten Hag’s “depends on the
opposition” and “how good is the team around him” comments sound like hints. In contrast to Inter’s Simone Inzaghi, the United manager may harbour doubts about
deploying Onana high up the pitch against City.

The Dutchman is still building his team and may be wary of the uneven form of United’s new keeper. Bought for £47.2m, Onana arrived on a ticket of being a
high-grade shot-stopper, technically impressive too with the ball at his feet. But until impressing against Brentford and Copenhagen, Onana had bombed, costing
United goals and points.
</item 94>

<item 95>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 23.36 GMT Jos Buttler’s reaction We probably didn’t adapt quickly enough with the bat – I thought 16o w0uld have been a decent score on a used wicket. It
was a fantastic bowling effort; we did well to hang in the game as long as we did. Phil Salt has had a brilliant series. We found out lots about ourselves as a
team. [On the stop-clock experiment] It was only in the last game that we really noticed it. It’s good to keep the game moving.



6d ago 23.19 GMT England’s miserable white-ball year ends with another series defeat. Plenty of good has come from this series, though, most notably the
performance of Phil Salt. While there are a few slots to fill, most notably at No3, they are in reasonable shape ahead of nexr year’s World Cup.
</item 95>

<item 96>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
The current tour of India is not a multi-format series, but the defeat in the one-off Test marks Australia’s first loss in a four-day game since January 2014.

But quizzed over whether the defeat suggested Australia’s unprecedented period of dominance was ending, Healy was defiant.

“It’s probably not even a blip,” Healy said.

Advertisement

“It (Test cricket) is a format where we’re not overly familiar with, in conditions that we’re not familiar with.

“I see more positives out of this and than any sort of negativity.

“There is always a lot of expectation and external noise about our team and how we’re performing.

“They are so used to seeing us winning, that I think, at times, they forget that we’ve actually seen quite a lot of change over the last 12 to 18 months.

“That kind of goes unnoticed external to our group.”

Australia have spoken about shifting their mindset in white-ball cricket after the losses in England, pursuing a more attacking approach.
</item 96>

<item 97>
Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see (The Roar | Sports Writers Blog,
2023-12-07T22:28:31+00:00)
“I thought we were poor in both areas with our finishing and also both goals we conceded were terrible.

There’s no point in feeling sorry for ourselves, looking for a cuddle anywhere. There’s only one way to change our circumstances and that is to come here on
Sunday and put in a performance, not just play good football but go out there and show some conviction about ourselves as a team.

Advertisement

“Sometimes we can disguise how we’re going by playing some nice stuff but like I said from day one, that’s not what I’m about.

“I want to win and that’s why I came to this football club and that’s the message. We’ve still got a long way to go, I’ve said that from the start, we’re still
right at the beginning what we need to create and days like today just give me further evidence and fuel of how much we need to do.”

James Ward-Prowse puts West Ham ahead against Tottenham ????
</item 97>

<item 98>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
That is all.”



6d ago 22.13 GMT 6th over: West Indies 39-2 (Charles 16, Hope 5) Charles mistimes a short ball from Woakes just short of Moeen in the covers. In the end Moeen
does well to save the boundary. This has been a good fightback by England, with only nine runs scored in the last three overs.



6d ago 22.09 GMT 5th over: West Indies 37-2 (Charles 15, Hope 4) Topley continues, which makes sense because the spinners will be doing most of the work after
the Powerplay, and concedes only three runs. He has quietly had a good series and is comfortably the most economical seamer on either side.
</item 98>

<item 99>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Just as surely as when they came out on Tuesday facing an almost unachievable task, the outcome of the West Indies’ run chase seemed inevitable from the start.
With such a meagre target they needed little more than to keep their heads, and though both Topley and Sam Curran conceded only two with their final overs to
push the contest into its final over, that never looked in doubt.

skip past newsletter promotion Sign up to The Spin Free weekly newsletter Subscribe to our cricket newsletter for our writers' thoughts on the biggest stories
and a review of the week’s action Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more
information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy
. We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Quick Guide How do I sign up for sport breaking news alerts?
</item 99>

<item 100>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.10 GMT 3rd over: West Indies 36-1 (Mayers 10, Hope 4) Shai Hope is the new man and digs out a Woakes yorker and somehow manages to crunch it away
down the ground for four. The replay of the Duckett catch is extremely enjoyable, Woakes owes him one for that as it was a poor ball outside off stump.



13 Dec 2023 00.07 GMT WICKET! Brandon King c Duckett b Woakes 22 (West Indies 32-1) Chris Woakes into the attack to replace the beleaguered Sam Curran. OH. MY.
LORDY. Ben Duckett flies to his left at backward point to pull off a spectacular diving catch off a full blooded cut shot from King. Stunner!
</item 100>
=== END ITEM id=dce612309a9b5bcd ===

=== ITEM id=4c216fba5400a2d0 ===
Gold sentences and facts:
Sentence 0: Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
  Fact 0.0: Kenneth Walker III plays for the Seahawks.
  Fact 0.1: Kenneth Walker III took four carries against the Rams.
  Fact 0.2: Kenneth Walker III left the game with an oblique injury.
Sentence 1: And Klopp has a new worry: the Reds are without Andy Robertson after the left-back suffered a shoulder injury while on international duty with Scotland.
  Fact 1.0: Klopp has a new worry.
  Fact 1.1: The Reds are without Andy Robertson.
  Fact 1.2: Andy Robertson is a left-back.
  Fact 1.3: Andy Robertson suffered a shoulder injury.
  Fact 1.4: Andy Robertson suffered the shoulder injury while on international duty with Scotland.

Retrieved items:
<item 1>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
In its privacy policy, the ChatGPT maker makes a passing acknowledgement of the objection requirements attached to relying on “legitimate interest” (LI),
pointing users towards more information about requesting an opt out — when it writes: “See here for instructions on how you can opt out of our use of your
information to train our models.”

What controversies have surrounded ChatGPT?

Recently, Discord announced that it had integrated OpenAI’s technology into its bot named Clyde where two users tricked Clyde into providing them with
instructions for making the illegal drug methamphetamine (meth) and the incendiary mixture napalm.

An Australian mayor has publicly announced he may sue OpenAI for defamation due to ChatGPT’s false claims that he had served time in prison for bribery. This
would be the first defamation lawsuit against the text-generating service.

CNET found itself in the midst of controversy after Futurism reported the publication was publishing articles under a mysterious byline completely generated by
AI.
</item 1>

<item 2>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
X owner Elon Musk last projected revenues of about $3 billion for this year. It’s anyone’s guess if that’s accurate now, but as an example, that would result in
a fine of up to $180 million.

Indirectly, an investigation will also be a key way for the public to have significantly more transparency around how X is being used (and abused): since Musk
took Twitter private, the company no longer produces quarterly earnings reports and is not held to account as closely as a result.

EU officials confirmed to TechCrunch that an investigation has not been taken at this point. Clearly, though, yesterday’s development (and the pace at which
regulators are moving) is a strong indication of the bloc’s direction of travel.

The Commission said it is looking at X’s compliance with the DSA across a number of areas — including with regard to its policies and practices regarding
notices on illegal content; complaint handling; risk assessment and measures to mitigate the risks identified.
</item 2>

<item 3>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Under Musk, X has pushed the use of an existing crowdsourced fact-checking feature (rebranded as Community Notes) which essentially seeks to outsource the issue
to a hands-length process of aggregating views from platform users, some of which may be appended as contextual notes to dubious tweets. At the same time the
erratic billionaire has set about removing signals Twitter users were previously able to rely on to help assess the quality of information tweets contained,
such as removing state-affiliated media labels from propaganda outlets and replacing legacy account verifications with pay-to-play.

X was contacted for a response to the EU’s analysis of the ratio of disinformation it’s spreading. It replied with an automated email reading: “Busy now, please
check back later.”
</item 3>

<item 4>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT got an overall three-star rating in the report, with its lowest ratings relating to transparency, privacy, trust and safety.

OpenAI blames DDoS attack for ChatGPT outage

OpenAI confirmed that a DDoS attack was behind outages affecting ChatGPT and its developer tools. ChatGPT experienced sporadic outages for about 24 hours,
resulting in users being unable to log into or use the service.

OpenAI debuts GPT-4 Turbo

OpenAI unveiled GPT-4 Turbo at its first-ever OpenAI DevDay conference. GPT-4 Turbo comes in two versions: one that’s strictly text-analyzing and another that
understands the context of both text and images.

GPT-4 gets a fine-tuning

As opposed to the fine-tuning program for GPT-3.5, the GPT-4 program will involve more oversight and guidance from OpenAI teams, the company says — largely due
to technical hurdles.
</item 4>

<item 5>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Researchers discover a way to make ChatGPT consistently toxic

A study co-authored by scientists at the Allen Institute for AI shows that assigning ChatGPT a “persona” — for example, “a bad person,” “a horrible person” or
“a nasty person” — through the ChatGPT API increases its toxicity sixfold. Even more concerning, the co-authors found having the conversational AI chatbot pose
as certain historical figures, gendered people and members of political parties also increased its toxicity — with journalists, men and Republicans in
particular causing the machine learning model to say more offensive things than it normally would.

The research was conducted using the latest version, but not the model currently in preview based on OpenAI’s GPT-4.

Y Combinator-backed startups are trying to build ‘ChatGPT for X’

YC Demo Day’s Winter 2023 batch features no fewer than four startups that claim to be building “ChatGPT for X.”
</item 5>

<item 6>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
But at the time of writing, Musk had engaged in a bit of a dance of his own with the news of the EU’s warning by responding to a critical tweet posted to X by
journalist Glenn Greenwald, who attacked the EU’s new “censorship law,” as he dubbed the DSA — which he claimed would be used to “punish X.”

In reply to Greenwald, Musk avoided expressing the same trenchant criticism of the EU’s approach but invited yet more relativism — writing: “Let the public hear
exactly what this disinformation consists of and decide for themselves.”

He then went on to sew doubt that disinformation is something that can be independently arbitrated at all, implying fact-checking is just a convenient exercise
for targeting different opinions — so essentially echoing Greenwald’s position — by claiming that “many times” the “official fact-checker” has been found making
false statements, before adding a rhetorically empty offer that “Maybe this is the case here, maybe not” for good, plausible deniability measure.
</item 6>

<item 7>
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
</item 7>

<item 8>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
A person can also specify their family size, so the text-generating AI can give responses about meals, grocery and vacation planning accordingly.

The FTC is reportedly investigating OpenAI

The FTC is reportedly in at least the exploratory phase of investigation over whether OpenAI’s flagship ChatGPT conversational AI made “false, misleading,
disparaging or harmful” statements about people.

TechCrunch Reporter Devin Coldewey reports:

This kind of investigation doesn’t just appear out of thin air — the FTC doesn’t look around and say “That looks suspicious.” Generally a lawsuit or formal
complaint is brought to their attention and the practices described by it imply that regulations are being ignored. For example, a person may sue a supplement
company because the pills made them sick, and the FTC will launch an investigation on the back of that because there’s evidence the company lied about the side
effects.

OpenAI announced the general availability of GPT-4

Starting July 6, all existing OpenAI developers “with a history of successful payments” can access GPT-4.
</item 8>

<item 9>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
The concerns pushed policymakers in Europe to mandate security assessments for any products using generative AI systems like ChatGPT, and over 20,000
signatories — including Elon Musk and Apple co-founder Steve Wozniak — to sign an open letter calling for the immediate pause of large-scale AI experiments like
ChatGPT.

But examples of ChatGPT abuse in the wild have been few and far between — so far.

With the launch of GPTs, OpenAI’s tool for building custom conversational, action-taking AI systems powered by OpenAI’s models, including the models
underpinning ChatGPT, ChatGPT could become more a gateway to a broader ecosystem of AI-powered chatbots than the end-all-be-all.

With GPTs, a user can train a model on a cookbook collection, for example, so that it can answer questions about ingredients for a specific recipe. Or they can
give a model their company’s proprietary codebases so that developers can check their style or generate code in line with best practices.
</item 9>

<item 10>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Instead, users will automatically be switched to models based on the prompt.

ChatGPT officially gets web search

OpenAI has formally launched its internet-browsing feature to ChatGPT, some three weeks after re-introducing the feature in beta after several months in hiatus.
The AI chatbot that has historically been limited to data up to September, 2021.

OpenAI integrates DALL-E 3 into ChatGPT

The integration means users don’t have to think so carefully about their text-prompts when asking DALL-E to create an image. Users will also now be able to
receive images as part of their text-based queries without having to switch between apps.

Microsoft-affiliated research finds flaws in GPT-4

A Microsoft-affiliated scientific paper looked at the “trustworthiness” — and toxicity — of LLMs, including GPT-4.
</item 10>
=== END ITEM id=4c216fba5400a2d0 ===
