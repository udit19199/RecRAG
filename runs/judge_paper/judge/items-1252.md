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

This file has 1 items.

=== ITEM id=987594b3de559c35 ===
Gold sentences and facts:
Sentence 0: India were playing against Australia in the second semi-final of the 2015 World Cup in Sydney on March 26.
  Fact 0.0: India played against Australia in the second semi-final of the 2015 World Cup.
  Fact 0.1: The India versus Australia match was held in Sydney.
  Fact 0.2: The India versus Australia match was played on March 26.
Sentence 1: It's crunch time for England in the World Cups as they face South Africa in the semi-final of the Rugby World Cup and in the ODI Cricket World Cup.
  Fact 1.0: It is crunch time for England in the World Cups.
  Fact 1.1: England faces South Africa in the semi-final of the Rugby World Cup.
  Fact 1.2: England faces South Africa in the semi-final of the ODI Cricket World Cup.

Retrieved items:
<item 1>
England thrash Scotland but cannot avoid Team GB Olympic heartbreak (The Guardian, 2023-12-05T21:56:19+00:00)
A showdown.

There will be questions about Scotland’s efforts, a defensive collapse at Hampden Park almost helping England towards earning a place for Team GB at the Paris
Olympics, but that would be unfair. The gulf in class between the European champions, ranked fourth in the world, and their neighbours to the north, ranked
23rd, is clear and the manager, Pedro Martínez Losa, has been unconvincing in charge.

There were two changes to the England side that came from two goals down to earn Friday’s late 3-2 win against the Netherlands that had kept hopes of
progression alive. Beth Mead was returned to the starting lineup for the first time since suffering an anterior cruciate ligament injury just over a year ago,
having proved her worth coming off the bench at half-time against the Dutch to insert some directness into England’s play, and Esme Morgan replaced Jess Carter.
</item 1>

<item 2>
When was the first Cricket World Cup? History, winners' list and tournament details ahead of 2023 men's ODI showpiece (Sporting News, 2023-10-01T23:10:05+00:00)
Current cricketing powerhouse South Africa was banned at the time from international sport due to its policy of racial segregation under the Apartheid system,
and thus did not take part in the first ODI World Cup.

The West Indies won the first tournament and defended the title in the second World Cup in 1979, while India's first taste of glory came when they clinched the
trophy in 1983 by defeating the West Indies by 43 runs in the final.

ODI World Cup full list of winners and finals

Year Host Winner Runner-up Result 1975 England West Indies Australia West Indies won by 17 runs 1979 England West Indies England West Indies won by 92 runs 1983
England India West Indies India won by 43 runs 1987 India & Pakistan Australia England Australia won by seven runs 1992 Australia & New Zealand Pakistan England
Pakistan won by 22 runs 1996 Pakistan,
</item 2>

<item 3>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.45 GMT 14th over: West Indies 94-3 (Hope 19, Rutherford 30) West Indies are rushing towards a series victory. Rutherford hooks the new bowler Sam
Curran for six more, and there are 11 from the over in total. Rutherford has 30 from 23 balls, Hope 19 from 28. They’re playing their roles to perfection.



6d ago 22.40 GMT 13th over: West Indies 83-3 (Hope 17, Rutherford 21) Rehan Ahmed returns, perhaps in the hope of tempting Rutherford into one big shot too
many. Instead it’s Shai Hope, who is happily playing the anchor role, who faces five of the six deliveries. England are running out of time.
</item 3>

<item 4>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
But the first real in-match statement came on 27 minutes as New Zealand-born Aki – perhaps the player of the World Cup so far who had made a tournament-high 61
carries, 33 dominant carries and 23 defenders beaten heading into the weekend – brilliantly jinked inside two defenders, fended off another and powered through
a gap to dive over the line.

Yet more All-Black class saw Ardie Savea go over in the corner after quick ball was shipped wide but with scrum-half Aaron Smith in the sin-bin for a deliberate
knock-on, Ireland narrowed the gap to 18-17 just before the break as another New Zealand-born Irish star, Jamison Gibson-Park, sniped from a lineout maul and
somehow stretched out for the try.

The All Blacks extended an incredible record of leading at half-time of their quarter-final in every single World Cup but that lead had never felt more fragile.
</item 4>

<item 5>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
New Zealand played some scintillating rugby in victory (AFP via Getty Images)

They led 13-0 and though they may not have realised, no team had ever squandered a lead that large in a World Cup quarter-final. Not that they would have
expected an easy ride from there.

They conceded a penalty straight from the restart as their kick chase was illegally impeded and Sexton duly slotted the three.

Ireland had made something of a statement during the Haka as their fans drowned it out with a rousing rendition of Fields of Athenry and the players formed a
figure of eight in memory of ex-international and Munster coach Anthony Foley, who tragically died in 2016, as they stared down their opponents.
</item 5>

<item 6>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
“Some of those cricketers on that park today would go in your best ever England white ball XI. It’s just been a bridge too far for most of them.

“What I don’t like is giving players a cop out… when we win the 50-over World Cup and the 20-over World Cup, aren’t they great? We’re brilliant. And when the
wheels come of it’s the structure. It’s the structure of English cricket, we’re a disgrace.

“We play 20-over cricket, we play 100-ball cricket, we don’t play enough 50-over cricket – how much 50-over cricket domestically has Virat Kohli played? Or
Heinrich Klaasen or anyone out here?

“That’s what’s made this great side over the last six years, travelling round the world playing T20 franchises. It’s such a lame excuse.”
</item 6>

<item 7>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 21.39 GMT The difference two days make Tuesday, Tarouba England 267-3, 19 sixes

Thursday, Taroura England 132 all out, five sixes

Updated at 21.45 GMT

6d ago 21.36 GMT WICKET! England 132 all out (Curran c Rutherford b Holder 12) Sam Curran clouts the next ball to long-off to end an increasingly miserable
England innings. They were 109 for four after 14 overs, and then the house fell down. West Indies need 133 to win the series.

Updated at 21.38 GMT

6d ago 21.33 GMT WICKET! England 132-9 (Rashid run out 1) Adil Rashid sacrifices himself, just like in the 2019 World Cup final, so that Curran can stay on
strike. Four balls remaining.
</item 7>

<item 8>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
talkSPORT will have updates from all the action in Paris as England take on the Springboks from 8:00pm

9 Owen Farrell will be looking to inspire England to victory against the Springboks Credit: Getty

Over in India, England's Cricket World Cup hopes hang by a thread after their latest shock defeat to Afghanistan.

They will be boosted by the news that Ben Stokes is fit for the match after he picked up a hip injury five days before the tournament.

But victory for England could scarcely be more crucial.

Having also lost to New Zealand in their opener, their chances of finishing in the top four and progressing to the semi-finals are looking dubious.

But South Africa, despite winning their first two games, suffered a shock defeat to the Netherlands in their most recent match - a sign for England that the
Proteas can be beaten.
</item 8>

<item 9>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
2003 World Cup, against Kenya

India were playing underdogs Kenya in the last-four stage match in Durban on March 20.

With Man-of-the-Match India skipper Sourav Ganguly scoring an unbeaten 111 runs in 114 balls and Man-of-the-Tournament Tendulkar scoring an 111-ball 83, India
comfortably scored 270 for four wickets in 50 overs.

In reply, Kenya were no match for Indian dominance, and barring their skipper Steve Tikolo, who top-scored with 56 runs, no Kenyan batsman could score more than
29 runs.

With Zaheer Khan taking 3 wickets in 9.2 overs for just 14 runs, India could easily beat Kenya by 91 runs.

Though India breezed into semi-finals easily, the tournament is also remembered for Kenya's best run in the World Cup as the team from a country with limited
cricket facilities and no proper grassroots structure could reach as far as semi-finals.
</item 9>

<item 10>
When was the first Cricket World Cup? History, winners' list and tournament details ahead of 2023 men's ODI showpiece (Sporting News, 2023-10-01T23:10:05+00:00)
Country World Cup titles Years won Australia 5 1987, 1999, 2003, 2007, 2015 India 2 1983, 2011 West Indies 2 1975, 1979 England 1 2019 Sri Lanka 1 1996 Pakistan
1 1992

MORE: Complete ICC Cricket World Cup schedule and results from India

A 50-over ODI Cricket World Cup

The 1987 tournament saw two prime changes in the way the ODI World Cups were organised — a shift of venue from England to the Indian subcontinent, and games
being played in 50 overs instead of 60.

The change in the number of overs was attributed to shorter daylight hours in Asia in comparison to England.

The 1987 tournament also had a new winner in Australia, who went on to defeat England by seven runs.
</item 10>

<item 11>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Shikhar Dhawan scored a blistering 117 runs off 109 balls, with Virat Kohli joining in with 82 runs off 77 balls. In the second innings, Indian bowlers bowled
out the Australian side within 316 runs, with Bhuvneshwar Kumar and Jasprit Bumrah picking up three wickets and Yuzvendra Chahal picking two as India won the
match by 36 runs.

2023, 5th Match, ICC Cricket World Cup, Chennai

The most recent clash between these two sides happened in this World Cup. Australia won the toss, chose to bat first, and posted a target of 200 runs as they
were bowled in 49.3 overs. David Warner scored 41 runs off 52 balls, and Steven Smith scored 46 runs off 71 balls in that innings.
</item 11>

<item 12>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England replacements: Theo Dan, Ellis Genge, Kyle Sinckler, Ollie Chessum, Billy Vunipola, Danny Care, George Ford, Ollie Lawrence.

The Springboks are unchanged from that epic quarter-final with France, having all played their part as Jacques Nienaber's side hung on in the second half. The
contribution of backs Kurt-Lee Arendse, Damian de Allende and Cheslin Kolbe saw all three cross for first-half tries, with Kolbe's coming off the back of a
charged-down conversion from France's Thomas Ramos. Preventing those two points proved crucial in the end.

The key contribution, though, came from Eben Etzebeth, who bulldozed his way over to score the only five-pointer of the second half and wrestle the game back
into South African hands.
</item 12>

<item 13>
England thrash Scotland but cannot avoid Team GB Olympic heartbreak (The Guardian, 2023-12-05T21:56:19+00:00)
If there was any doubt about Scotland’s intentions against England, with Bronze calling accusations that Scotland players might feel conflicted with the
possibility of Olympic qualification on the line for them as much as for England “the rudest thing I’ve ever heard” and Scotland’s captain, Rachel Corsie,
calling it “disrespectful” and “outrageous” those were put to bed, temporarily at least, in the opening three minutes, with Bronze clattered into late twice as
Scotland looked to make an early impression.

If England had played well for the opening 10 minutes against the Dutch only to be sucker-punched on the counter in the 12th minute, against Scotland there was
no risk of that. Instead, the visiting team took the lead to swing the pressure on to the Dutch, Mead’s corner sent in by an unmarked Greenwood.

Moments later Lisa Evans was down in the box under pressure from Bronze but appeals for a penalty were waved away with the right-back having not caught the
forwards legs.
</item 13>

<item 14>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Australia batted first and posted a target of 261 runs courtesy of a 104-run knock of 118 balls by their captain, Ricky Ponting, with the loss of six wickets in
the process.

In the second innings, Sachin Tendulkar, Yuvraj Singh, and Gautam Gambhir scored three half-centuries as India changed down the target within 47.4 overs, losing
5 wickets in the process.

India went all the way to lift their second ODI World Cup trophy, defeating Sri Lanka in the finals.

2015, 2nd Semi-Final, ICC Cricket World Cup, Sydney

Australia got their revenge for the 2011 quarterfinals in this World Cup as they defeated India by 95 runs. Australia posted a target of 329 runs, losing seven
wickets in the process.
</item 14>

<item 15>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
It would have been four but for an outstanding stop from Rehan Ahmed.



6d ago 23.12 GMT 19th over; West Indies 124-6 (Hope 37, Holder 1) Holder is almost run out first ball! He took a very dodgy single to mid-off and was well short
of the crease when Curran just failed to gather Rashid’s throw. Nine to win from six balls, Chris Woakes to bowl them.



6d ago 23.10 GMT WICKET! West Indies 123-6 (Russell c Livingstone b Curran 3) Oh my. Russell slugs a low full toss straight down the throat of Livingstone at
long-on, and England – who have been behind the game throughout this innings – are again dreaming of an improbable victory. West Indies need 10 from 7 balls.
</item 15>

<item 16>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
Sarina Wiegman talks a lot about defending against counterattacks, because this is basically the key to the whole enterprise. Cut off the counter, and with
England’s technical quality they can easily dominate. It is also the essential measure of how the team is functioning as a collective: the efficiency of the
press, the closing of the angles, the command of space. Lineth Beerensteyn’s two goals were the sort England simply were not conceding four months ago.

Mary Earps shows her frustration after conceding a second goal to the Netherlands. Photograph: Tom Jenkins/The Guardian

So England passed the ball around a bit and gave it away a bit and essentially ran around trying to feel normal again, and on countless occasions Kelly or Hemp
would win a first or second ball, look up and see a penalty area empty of white shirts.
</item 16>

<item 17>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England and South Africa face off in the 2023 Rugby World Cup's second semi-final, played at the Stade de France in Saint-Denis, France. The game kicks off at 9
p.m. local time on Saturday, October 21, 2023.

Here are the kick-off times for this game around the world:

Location Date Time USA Sat, Oct. 21 3:00 p.m. ET Canada Sat, Oct. 21 3:00 p.m. ET Australia Sun, Oct. 22 6:00 a.m. AEDT UK Sat, Oct. 21 8:00 p.m. BST New
Zealand Sun, Oct. 22 8:00 a.m. NZDT India Sun, Oct. 22 12:30 a.m. IST Singapore Sun, Oct. 22 3:00 a.m. SGT Malaysia Sun, Oct.
</item 17>

<item 18>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The #Springboks are gearing up for another tough challenge this week with a semi-final clash against England 🇿🇦🏴󠁧󠁢󠁥󠁮󠁧󠁿#StrongerTogether #RWC2023
pic.twitter.com/HLSnnjWggB — Springboks (@Springboks) October 18, 2023

England are the only side still competing who are yet to lose a game, and that will stand them in good stead coming into this semi-final. Dominant wins over
Argentina, Japan and Chile sealed passage to the last eight, and after being given an almighty scare by Samoa in their final pool match, England overcame a
Fijian fightback in Marseille to triumph 30-24. Despite being outscored by three tries to two, captain Owen Farrell kicked 20 points, sending his side into the
last four at the expense of 2023's breakout stars.
</item 18>

<item 19>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
A wicket once so true turned untrustworthy and they struggled to a score of 132, less than half the 267 to which they had joyously motored just 48 hours
previously. And though the home side’s response was anything but overwhelming, neither was the result ever in significant doubt.

“We’ve had some high-scoring games, but the last game was almost the complete opposite of what happened here,” Reece Topley said. “We were probably just under
par with the first-innings score but with the ball we were excellent, pushed them right to the end. I think it’s been an amazing series. Both teams have played
some unbelievable cricket. I was so excited turning up today, because it was basically a final and those are the games you want to play in and be on the right
side of. The bottom line is you want to win this series, I want to win every series I can for England.”
</item 19>

<item 20>
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
</item 20>

<item 21>
When was the first Cricket World Cup? History, winners' list and tournament details ahead of 2023 men's ODI showpiece (Sporting News, 2023-10-01T23:10:05+00:00)
Both the host teams met in the semi-finals at Eden Gardens, Kolkata. India's poor performance sparked an unruly crowd to protest against the home team and Sri
Lanka was declared the winner by default, due to the halt in play.

Sri Lanka then went on to win the trophy that year, and they were back to try and defend it three years later in England, rather than the usual four-year cycle.

Following the 1996 event the ICC opted to try and revamp the schedule, with three years between events rather than four. But the move proved unpopular and 1999
was the only tournament to follow that pattern.

When was the last time India hosted the ODI Cricket World Cup?

The last time India welcomed the ODI Cricket World Cup was in 2011, when they were co-hosts with Sri Lanka and Bangladesh.

Pakistan were barred from hosting due to terrorist attacks on the Sri Lankan team when touring the country a couple of years prior to the World Cup.
</item 21>

<item 22>
England thrash Scotland but cannot avoid Team GB Olympic heartbreak (The Guardian, 2023-12-05T21:56:19+00:00)
That rare foray forward was urged on by a vocal Scotland crowd, in which interesting conversations could be heard. After Lauren Hemp raced free on the right a
hopeful “Come on Lauren” was met with a thick, cheery Scottish “Shut up!” behind the press box.

If it was friendly in the stands it was far less so on the pitch. England went close to a second, with the Dutch having taken the lead against Belgium, when
James weaved into the box before delivering to Hemp at the far post who lashed her effort off the base of a post from close range.

skip past newsletter promotion Sign up to Moving the Goalposts Free weekly newsletter No topic is too small or too big for us to cover as we deliver a weekly
roundup of the wonderful world of women’s football Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside
parties. For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Privacy Policy .
</item 22>

<item 23>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.37 GMT 12th over: West Indies 80-3 (Hope 15, Rutherford 20) England appeal unsuccessfully for caught behind when Rutherford misses a swipe at Rashid.
There’s only one review left and they decide not to risk it. A good job too; there was nothing on UltraEdge. Out of nothing Rutherford launches a full ball from
Rashid down the ground for six, almost knocking himself off his feet in the process. West Indies are managing an awkward chase pretty well.
</item 23>

<item 24>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
India, winning the toss, chose to bat first but crumbled within 125 runs in 41.4 overs. Brett Lee and Jason Gillespie picked up three wickets each for
Australia.

Matthew Hayden and Adam Gilchrist gave Australia a fiery start in the second innings as they reached the target in 22.2 overs, losing just one wicket. Australia
won the match by nine wickets, with 166 balls remaining.

2003, Final, ICC World Cup, Johannesburg

A memory that is still fresh in the minds of many Indians is the final of the 2003 World Cup. India, winning the toss, chose to bowl first, which turned out to
be a big mistake as Australia posted a target of 360 runs in 50 overs. Australian captain Ricky Ponting played a brilliant knock of 140 runs in 121 balls, along
with Damien Martyn, who scored 88 runs in 84 balls.
</item 24>

<item 25>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
2023 World Cup, against New Zealand

To be played
</item 25>

<item 26>
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
</item 26>

<item 27>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Amid the wider theme of transformation Adil Rashid’s excellence remains constant. The 35-year-old, who rose to the top of the T20 bowling rankings for the first
time on Wednesday, took two wickets and conceded just 21 while Topley was even better, also claiming a pair of victims but conceding only 17. England take
plenty of positives from this series, but the result was not one.
</item 27>

<item 28>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Mohammad Azharuddin and Maninder Singh picked up three wickets for India.

1992, Match 12, Benson & Hedges World Cup, Brisbane

The game in which the rain spoiled not just the match but India's winning chances as they lost by just one run as revised target. Australia set a target of 237
runs in 50 overs, losing nine wickets in the process. Dean Jones scored 90 runs off 108 balls for Australia. Kapil Dev and Manoj Prabhakar picked up three
wickets each for India. In the second innings, rain interrupted play after 16.2 overs. The revised target was recalculated to 236 off 47 overs. India needed 5
runs off 4 balls. However, India was bowled out due to two run-outs on the very last ball. Mohammad Azharuddin scored a brilliant 93 runs off 102 balls for
India.
</item 28>

<item 29>
England thrash Scotland but cannot avoid Team GB Olympic heartbreak (The Guardian, 2023-12-05T21:56:19+00:00)
Martínez Losa swapped four players from the side that earned a 1-1 draw with Belgium on Friday, with Lee Alexander notably replacing the former England
international Sandy MacIver in goal.

Lauren James scores England’s third goal in their emphatic win. Photograph: Carl Recine/Reuters

Wiegman had urged her players to “go wild from the first minute” against Scotland, cautioning “but you don’t want to get erratic” and England came out of the
blocks searching for an early goal that would assert their authority on the tie.
</item 29>

<item 30>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India were playing their sixth semi-finals at a place where they played their first World Cup semi-final in 1983, Manchester. However, with a different team-
New Zealand.

Indian bowlers raised hopes when they restricted New Zealand to 239 for 8 in 50 overs with Bhuvneshwar Kumar taking three wickets.

Ross Taylor 74 runs (90 balls) and skipper Kane Willamson 67 runs (95 balls) were the top scorers for New Zealand. But Kiwi Man-of-the-Match bowler Matt Henry
sent openers KL Rahul and Rohit Sharma back to the pavilion soon, followed by the dismissal of Dinesh Karthik in the 10th over of India's innings.

With the Indian top-order failing, Ravindra Jadeja (77 runs in 59 balls) and wicketkeeper MS Dhoni (50 runs in 72 balls) tried to steady the Indian ship.

But once they fell, it was all over for India as they lost the battle by 18 runs.
</item 30>

<item 31>
When was the first Cricket World Cup? History, winners' list and tournament details ahead of 2023 men's ODI showpiece (Sporting News, 2023-10-01T23:10:05+00:00)
The 2011 World Cup was a tournament of many firsts: Australia's streak of remaining unbeaten in 35 World Cup matches was broken when they were defeated by
Pakistan. They hadn't lost a game since the 1999 tournament.

It was also the first time two Asian teams were in the finals, as India became the first country to win a World Cup at

home by defeating Sri Lanka in the decider.

After a prolonged drought of ICC trophies, India are favourites to bring the trophy home again this time in 2023.
</item 31>

<item 32>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
Pressure? The South African laughs at the notion of what pressure is compared to the inequality and challenges the Rainbow Nation continues to experience.

Advertisement

The question is: how can New Zealand’s All Blacks withstand and overcome the hitherto unshakable wave of momentum and spirit the Springboks play with under
Erasmus in this weekend’s World Cup final?

It would seem that, as the Springboks play for their country, the All Blacks are playing for themselves as much as anyone – and the generation of players and
leaders that will call time on their international careers at week’s end.

The backlash and toxicity that followed their historic home series loss to Ireland was a humbling and public reminder of the expectation of being an All Black.

It’s why Saturday’s World Cup final is personal for the men wearing black.

All Blacks social media managers were forced to delete a thousand comments an hour following the humbling series loss in Wellington.
</item 32>

<item 33>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Damien Fleming took five wickets for Australia as Australia won the match by 16 runs.

1999, 1st Super, ICC World Cup, The Oval

The Australian dominance continued in this World Cup as well, as they defeated India by 77 runs. Australia posted a target of 283 runs in 50 overs courtesy of a
brilliant 83 of 99 balls by Mark Waugh.

India had an early batting collapse but was stabilised by a sensible 100 off 138 balls by Ajay Jadeja. Robin Singh also scored 75 runs off 94 balls. However,
India was bowled out within 205 runs in 48.2 overs. Glenn McGrath picked up three wickets for Australia, with Damien Fleming and Steve Waugh picking up two
each.

2003, Match 11, ICC World Cup, Centurion

This was India's biggest batting collapse in the history of the ODI World Cup.
</item 33>

<item 34>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“It’s the life of this business, it’s a tough game. It’s tough when you’re trying to get your performance right. It’s a tough game when people around you see
things differently, but we’ve learned a lot about ourselves.

“We take massive pride in the jersey and making sure we reach the levels we want to. The rest of that is probably a conversation for another day.

“It hasn’t altered our vision of where we want this team to be. I know we’ve surprised a few people but I don’t think we’ve surprised ourselves.”
</item 34>

<item 35>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
STADE DE FRANCE – Twenty-eight years between drinks and drama was always on the cards between these two great foes. Little did we know just how much.

But just like that famous win at Ellis Park where Nelson Mandela handed over the William Webb Ellis Cup to Francois Pienaar, it was the Springboks who once
again prevailed 12-11 in an instant classic.

Unlike the blue skies that greeted Johannesburg and the Boeing 747 plane that sent shivers down the spines of everyone in attendance in 1995, rain and dampness
poured down on the Stade de France.

Facebook Twitter Whatsapp Reddit Email Share

Yet, nothing was going to dampen this evening as 80,065 fans poured in.

In a match of controversy, where cards of yellow and red were handed out quicker than a game of Uno, thundering shots both legal and illegal were put on by
rugby’s two greatest sides.
</item 35>

<item 36>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The last time Cheslin Kolbe played England at a Rugby World Cup... 🇿🇦 🏆#RWC2023 | #ENGvRSA pic.twitter.com/Riw2uaSaKa — Rugby World Cup (@rugbyworldcup) October
19, 2023

Both sides' journeys have seen twists, turns, and bumps in the road, but it's led them both to within 80 minutes of a Rugby World Cup final. England will no
doubt be looking to echo the spirit of 2019, when they beat the All Blacks 19-7 to reach the final, but in South Africa, they face a side who have taken their
game to new heights just to reach the semis.

MORE: 2023 Rugby World Cup final: Details, kick-off time and every team's route

What time does England vs South Africa kick off?
</item 36>

<item 37>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
Four years ago, Rassie Erasmus delivered one of the great rugby speeches ahead of the 2019 World Cup final.

“You don’t have the right today to worry about your mistakes,” Erasmus said before the Springboks sealed a stunning 32-12 victory over England to claim their
third William Webb Ellis Cup.

“If you worry about your mistakes today, you’ve got ego problems.

Facebook Twitter Whatsapp Reddit Email Share

“You are not representing yourself today. You are fighting for the things that happen in South Africa.

“You are tackling for the boy who didn’t get the same opportunities that you got eventually.”

Four years later and the “ego” Erasmus spoke about is what is driving the standards and expectations around the Springboks’ bid to go back-to-back.

“We explain to the whole squad why we are going a certain route.
</item 37>

<item 38>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The latter is winning only his seventh England cap but gets the starting nod alongside Maro Itoje, with Genge, Chessum and the rest of the replacements hoping
to make an impact sizeable enough to overcome the Springboks' infamous 'bomb squad'.

Their time is now 🙌



Here is the @EnglandRugby team to take on reigning champions South Africa in the semi-final in Paris#RWC2023 | #ENGvRSA pic.twitter.com/nkf6rXBPUU — Rugby World
Cup (@rugbyworldcup) October 19, 2023

England starting XV: Freddie Steward, Jonny May, Joe Marchant, Manu Tuilagi, Elliot Daly, Owen Farrell, Alex Mitchell; Joe Marler, Jamie George, Dan Cole, Maro
Itoje, George Martin, Courtney Lawes, Tom Curry, Ben Earl.
</item 38>

<item 39>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Also Read: IND vs AUS ICC World Cup 2023 Awards Winners List: ICC Player of the Match, ICC Player of the Tournament, Golden Bat and Golden Ball

Here are all the details you need to know about India vs Australia head-to-head in the ODI World Cups—

1983, Match 11, Prudential World Cup, Nottingham

India met with Australia for the very first time in this World Cup and lost the match by 162 runs. Australia posted a target of 321 runs courtesy of a brilliant
110 runs off 131 balls by Trevor Chappell. Kapil Dev took a fifer in that innings. India crumbled within 158 runs, with Ken MacLeay picking up six wickets for
Australia.

1983, Match 23, Prudential World Cup, Chelmsford

India got their revenge in the very same World Cup, as they defeated the Australian side by 118 runs.
</item 39>

<item 40>
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
</item 40>

<item 41>
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
</item 41>

<item 42>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
And they boast a brilliant record against the Blues of late, winning six out of their last seven meetings.

They have triumphed in their last three trips to Stamford Bridge, and haven't lost in a London derby since May 2022.

As for Chelsea, they go into their match against the Gunners hopeful of a result having won their last three games in all competitions.

Both teams face injury doubts, with Arsenal to potentially be without Bukayo Saka and William Saliba, while it remains to be seen if Reece James and Armando
Broja feature for the hosts.

9 Saka missed Arsenal's 1-0 victory over City at the Emirates last time out Credit: Getty

Sheffield United desperate for points against lacklustre Manchester United

With just one point on the board, Sheffield United's season could not be going much worse given they are rock bottom.

With a fixture against Manchester United on the cards, though, they may feel their hopes could be looking up given Erik ten Hag's men disappointing season.
</item 42>

<item 43>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
They moved into the last-four stage on the next two occasions when the World Cup was held in India, in 1987 and 1996.

India were in semi-finals in 2003 in South Africa.

Eight years later, they were again in last four at the 2011 World Cup as hosts, and repeated their feat in 2015, 2019, and 2023.

It means that in their last six World Cups, India have been in the semi-finals on five occasions.

Also Read: ICC Cricket World Cup 2023 semi-final schedule: India to play New Zealand in Mumbai; South Africa to face Australia in Kolkata; know venue, squads

In this write-up, ZeeBiz will take you through the memory lane of India's semifinal journey in the biggest cricket extravaganza in the world.
</item 43>

<item 44>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
Click below for more ⤵️ https://t.co/LeUuw4kPfP — RugbyPass (@RugbyPass) October 19, 2023

Marcus Smith hardly put a foot wrong at fullback against Fiji, but the world champions are an entirely different proposition for England to deal with at the
back. The high-ball security and hulking frame of Freddie Steward returns to the No. 15 jersey in one of three changes, while at inside centre, Manu Tuilagi
starts hoping to repeat his heroics of four years ago, when his early try in a dominant semi-final performance took England past the All Blacks.

England's other two changes come in the tight five, with Steve Borthwick making some big calls ahead of this game. Joe Marler starts ahead of vice-captain Ellis
Genge at loosehead prop, and Ollie Chessum is replaced in the second-row by George Martin.
</item 44>

<item 45>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
talkSPORT will bring you updates as England take on South Africa in the Cricket World Cup in Mumbai from 9:30am

9 It's not looking good for England's cricketers

UFC 294 in Abu Dhabi

Islam Makhachev is hoping to to successfully defend his UFC lightweight belt against Alexander Volkanovski at UFC 294.

Makhachev retained his title in February this year as he defeated Volkanovski by a unanimous decision.

But despite the Russian's favourable 24-1-0 record, Volkanovski will be desperate to add the lightweight belt to his featherweight title.

Also on the cards, Kamaru Usman will fight Khamzat Chimaev in a middleweight title eliminator.

Expect fireworks.
</item 45>

<item 46>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
India fought well, courtesy of two well-made 70s by Krishnamachari Srikkanth and Navjot Singh Sidhu. Unfortunately, India was bowled out in 49.5 overs with 269
runs on the board. Craig McDermott took four wickets for Australia.

1987, Match 15, Reliance World Cup, Delhi

This is the second time India has met Australia in this World Cup, with India winning this one by 56 runs. India batting first set a target of 290 runs courtesy
of four half-centuries by Sunil Gavaskar, Navjot Singh Sidhu, Dilip Vengsarkar, and Mohammad Azharuddin. Australia was bowled out within 233 runs, with David
Boon scoring 62 off 59 balls. Mohammad Azharuddin and Maninder Singh picked up three wickets for India.
</item 46>

<item 47>
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
</item 47>

<item 48>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The Springboks have done it tough, too, having already faced two of the tournament favourites on their way to the semi-finals. Comfortable defeats of Scotland,
Romania and Tonga couldn't mask the disappointment of the 13-8 loss to Ireland, which consigned South Africa to second in Pool B. However, with Ireland sent
home by the All Blacks in the quarters, the Bokke showed their World Cup mettle to edge the hosts 29-28 at the Stade de France, sneaking past Les Bleus in one
of the best games of rugby you're ever likely to see.
</item 48>

<item 49>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
They’ll always have Paris. Throughout this Nations League campaign this has been an England team fighting its own sense of entropy, still giving everything,
still trying the same plans and processes in the hope that eventually something would click. And at 9.34pm on a briskly cold Wembley night, it finally did. Ella
Toone’s injury-time goal earned England three points and kept their faintly glowing Olympic dream alive. But somehow it also did so much more.

It breathed life into a team and a cycle, and a moment that for so much of the evening had felt like it was drawing to its natural conclusion provided a rousing
encore to a cast preparing for its final curtain. There will be new challenges and new frontiers to conquer soon enough, a fresh set of European Championship
qualifiers in the spring, but for now this iteration of the Lionesses still has tricks up its sleeve, still has talent and creativity to burn, still has a fire
that can scorch any team on the planet.
</item 49>

<item 50>
England thrash Scotland but cannot avoid Team GB Olympic heartbreak (The Guardian, 2023-12-05T21:56:19+00:00)
For more information see our Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after
newsletter promotion

Eight minutes later they doubled their lead. Mead’s corner was cleared and sent back in by the Arsenal forward only to be cleared again. But while England
players appealed for handball, James sent the ball goalward and it took a heavy deflection that wrongfooted Alexander and flew in.

It wasn’t pretty, but England were clawing away at the goal difference gap. A minute later James delivered in more characteristic style, curling a wonderful
effort into the top corner from the edge of the area.

In added time at the end of the half, James turned provider of the goal that put England top of the group once more, sweeping a cross to Mead at the far post
for the forward to bring down and fire past a helpless Alexander.
</item 50>

<item 51>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
Top 10 Bowlers of ODI World Cup 2023:

Rank Player Mat Wkts 1 Mohammed Shami (IND) 7 23 2 A Zampa (AUS) 11 23 3 D Madushanka (SL) 9 21 4 JJ Bumrah (IND) 11 20 5 G Coetzee (SA) 8 20 6 Shaheen Shah
Afridi (PAK) 9 18 7 M Jansen (SA) 9 17 8 RA Jadeja (IND) 11 16 9 Josh Hazlewood 11 16 10 MJ Santner (NZ) 10 16 11 BFW de Leede (NED) 9 16

*Still Bowling

10.
</item 51>

<item 52>
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
</item 52>

<item 53>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
“Morgan’s England team was a finely tuned machine with everyone knowing their role and most of them at the top of their form… all Morgan had to do was pull the
strings of a great unit of cricketers and he did it well, but my Mum could have captained that lot.

“On the other hand Jos has a dysfunctional squad of players. Poorly selected, poorly prepared, not a settled team, many not sure of their roles, patchy form,
confidence wobbly and to cap it all England’s best two and most influential white-ball players have not been available.”

Boycott’s voice was added to by former England Test captains Nasser Hussain and Michael Atherton, who have both decried the sudden decline of the team’s
previously dominant white-ball set-up, having entered the World Cup as reigning ODI and T20I champions.
</item 53>

<item 54>
When was the first Cricket World Cup? History, winners' list and tournament details ahead of 2023 men's ODI showpiece (Sporting News, 2023-10-01T23:10:05+00:00)
ODI World Cup kits, white ball introduction

Held in Australia and New Zealand, the 1992 World Cup saw a number of landmark moments.

That tournament was the first in which teams wore coloured kits, while it also saw the introduction of the white ball, changes in fielding rules, day/night
matches and the inclusion of South Africa in the event for the first time following the end the Apartheid regime.

After a slow and disappointing start to their year, Pakistan made an historic comeback in the tournament to eventually win the trophy by defeating England in
the final.

READ: ICC sends 2023 World Cup trophy into space

The 1999 World Cup and a three-year gap

The 1996 tournament was jointly hosted by India and Sri Lanka, and will go down in history as one of the most controversial episodes in the game.

Both the host teams met in the semi-finals at Eden Gardens, Kolkata.
</item 54>

<item 55>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
The All Blacks are 18-point favourites for the opening semi-final.

It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.

“The past sets us up beautifully for both teams,” Foster said.

“We do know each other but we don’t know each other terms of a Rugby World Cup, so at this stage in the tournament it’s new territory for us and as we’ve seen
World Cups are very different.”

“We have learnt to greatly respect Argentina. They have a rich history of overachieving at the World Cup and have done a fantastic job to get here at the same
level we are, so it is going to be a heck of a game.”

Advertisement

Foster said the All Blacks, who suffered a shock semi-final loss to Eddie Jones’ England in 2019, never consider themselves favourites.

“Firstly, you’ve never heard us say we’re favourites,” he said.
</item 55>

<item 56>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Also Read: ICC Cricket World Cup 2023 Prize Money: How much finalists, semi-finalists, and other teams will win at ODI World Cup

Also Read: ICC ODI Cricket World Cup 2023, Winners List from 1975 to 2023: A look back to the history of the ODI World Cup

Plenty of wounds are still to be healed for India, as the defeat in the 2003 final against Australia still feels fresh. It is only time to tell if the Aussies
will repeat history or if it will be India who will write history by winning their third ODI World Cup.
</item 56>

<item 57>
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
</item 57>

<item 58>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
1. Mohammad Shami, India

Shami announced his return to the World Cup with a bang, picking a fifer in the very first match he played for India. He has picked up 24 wickets so far in just
six matches, conceding just 257 runs. He has picked up three five-wicket hauls including a seven-wicket haul against New Zealand in the semi-final on Wednesday,
November 15 at the Wankhede Stadium in Mumbai.

Catch the latest stock market updates here. For all other news related to business, politics, tech, sports, and auto, visit Zeebiz.com.
</item 58>

<item 59>
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
</item 59>

<item 60>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India, one of the favourites to win the World Cup, were very much on the path to victory till Mohammad Azharuddin, who top-scored with 64 runs was at the
crease.

But with the fall of his wicket, the Indian lower order collapsed like a house of placards as India lost the match by 35 runs.

1996 Wills World Cup, against Sri Lanka

The 1996 Cricket World Cup is known for Sri Lanka's emergence as a cricketing power on the horizon of world cricket and the beginning of Sachin Tendulkar's
extraordinary dominance in batting history.

Playing in the first semi-final at the Eden Gardens in Kolkata on March 13, India were favourites again, and they raised their hopes when they managed to
restrict Sri Lanka to 251 for 8 in 50 overs despite Man-of-the-Match Aravinda de Silva's 66 and Roshan Mahanama's 58.
</item 60>

<item 61>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Mohammad Azharuddin scored a brilliant 93 runs off 102 balls for India.

1996, Match 19, Wills World Cup, Mumbai

This match can be considered one of the most exciting matches played between these two sides. Australia batted first and posted a target of 259 runs in 50
overs, courtesy of a brilliant 126 runs off 135 balls by Mark Waugh. Venkatesh Prasad and Venkatapathy Raju picked up two wickets each for India.

India batted well courtesy of Sachin Tendulkar, who made a brilliant 90 runs off 84 balls, along with Sanjay Manjrekar, who scored 62 runs off 91 balls.
However, India was bowled out within 242 runs in 48 overs. Damien Fleming took five wickets for Australia as Australia won the match by 16 runs.
</item 61>

<item 62>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India seamer Javagal Srinath took three top-order wickets, including Sanath Jayasuriya and Romesh Kaluwitharana.

Fans at the Eden Garden were cheering loudly till Tendulkar was at the crease.

But once he fell for 65 with the team at 98, India suffered one of the worst collapses in the batting history of the World Cups.

The crowd turned angry and hostile towards Sri Lanka, creating a ruckus at the stadium, and the match was interrupted.

With no respite from onlookers' side, umpires decided to call off the match when India were reeling at 120 for eight in 34.1 overs.

The long-lasting memories of the crowd's bad behaviour and a weeping Vinod Kambli returning to the pavilion are still fresh in the memories of the cricket fans
of that generation.
</item 62>

<item 63>
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
</item 63>

<item 64>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
This innings, by Ponting, had famously created the conspiracy that he had Spring in his bat.

India gave up a tough fight, with Virender Sehwag making 82 runs off 81 balls and Rahul Dravid also scoring 47 runs off 57 balls. However, the lack of support
from the other batters led to India being bowled out within 234 runs. Brett Lee and Andrew Symonds took two wickets each as Australia lifted the World Cup for
the third time.

Also Read: India vs Australia, World Cup final in 2003 vs 2023: Can India change the course of history in the final?

2011, 2nd Quarter-Final, ICC Cricket World Cup, Ahmedabad

After so much dominance by the Australian side in the World Cups against India, India finally got their revenge in this World Cup as they defeated them by five
wickets.
</item 64>

<item 65>
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
</item 65>

<item 66>
England thrash Scotland but cannot avoid Team GB Olympic heartbreak (The Guardian, 2023-12-05T21:56:19+00:00)
An anxious wait, then complete heartbreak. No one could question the effort, the heart or the drive of Sarina Wiegman’s Lionesses in this game, but scoring six
goals against Scotland was not enough to close the goal difference gap on the Netherlands, because the Dutch scored a fourth against Belgium deep in added time
to secure top spot and progress from their Nations League group ahead of England.

It was an extraordinary mountain and a Lauren James masterclass and gut-busting display from Lucy Bronze were not enough to lift them to the summit. The damage
had been done in the preceding games, two poor goals conceded against the Dutch at Wembley, a 3-2 defeat in Belgium in October and a 2-1 loss to the Netherlands
in September, will be full of regrets. However, there will be what‑ifs either way, the first goal conceded in Utrecht appearing offside in the buildup.

It meant a back and forth of goals scored by England against Scotland and the Netherlands in their win over Belgium was necessary in the final round of
fixtures.
</item 66>

<item 67>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England vs. South Africa betting odds, prediction

The last time these two sides met was a tumultuous period to say the least, particularly for English rugby. South Africa's dominant 27-13 victory at Twickenham
proved to be the last game in charge for Eddie Jones, the coach who had taken England to the 2019 World Cup final. In his place, Steve Borthwick was hired, and
he's just one game away from repeating that feat.

England's form at this tournament has been perfect — at face value. Argentina, Japan, Chile, Samoa and Fiji have all provided highlights — Los Pumas are also in
the semi-finals — and England have beaten all of them. However, only Japan and Chile were defeated with bonus points, and the narrow wins over both Samoa and
Fiji in their last two games showed that England are certainly gettable.
</item 67>

<item 68>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 21.31 GMT 19th over: England 129-8 (Curran 9, Rashid 1) The hat-trick ball is another attempted yorker that Rashid flicks wide of Russell for a single.
He would have had three wickets in four balls had Hosein taken a tricky low chance at third man to dismiss Curran. Russell, who bowled really well, finishes
with two for 25.
</item 68>

<item 69>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
They sit tenth in the table having won four and lost as many from their opening eight games.

But their most recent Premier League victory over Brentford may give the Red Devils hope their season is on the up.

Sheffield United vs Manchester United is live on talkSPORT - kick off is at 8:00pm

9 United got a dramatic win over Brentford before the international break Credit: Getty

An England vs South Africa double header

If you've had your football fix for the day, there's also a packed Saturday of rugby and cricket.

It's crunch time for England in the World Cups as they face South Africa in the semi-final of the Rugby World Cup and in the ODI Cricket World Cup.

After England secured victory over Fiji in the quarter-finals, they now look to overcome defending champions South Africa to advance into their second
consecutive Rugby World Cup final.
</item 69>

<item 70>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
IND vs AUS Head-to-Head Record in ODI World Cup: It feels like a call back from the past, as two decades ago, these two goliaths—India and Australia—of the
cricketing world faced each other in the final of the ICC Men's ODI World Cup in Johannesburg, South Africa.

Also Read: India vs Australia World Cup Final Free Live Streaming: When and how to watch ICC World Cup 2023 Final IND vs AUS match live on Mobile apps, tv,
laptop, online

Once again, we are at the same stage as the two-time World Champions India will be locking horns against the five-time World Champions Australia for the finale
of the ICC Men's ODI World Cup 2023 on Sunday, November 19, at the Narendra Modi Stadium in Ahmedabad.
</item 70>

<item 71>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
For India, Ravindra Jadeja picked up three wickets, along with Kuldeep Yadav and Jasprit Bumrah picking up two each.

India was in a terrible position at the start of the second innings as both of the openers got out for a duck, and India lost three wickets for just two runs.

However, two brilliant and sensible innings by KL Rahul and Virat Kohli, where they scored 97 runs off 115 balls and 85 runs off 116 balls, took India to shore
as India won the match by six wickets.

Also Read: 2023 ICC World Cup: Virat Kohli vs Australia; know the ODI batting records and stats

Catch the latest stock market updates here. For all other news related to business, politics, tech, sports, and auto, visit Zeebiz.com.
</item 71>

<item 72>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
He and Franco Mostert pack down together in the second row, while hooker Bongi Mbonambi, utility Deon Fourie, veteran Duane Vermeulen and the devilish Kwagga
Smith will all be looking to back up superb quarter-final performances to take the Springboks to a second consecutive World Cup final.

South Africa starting XV: Damian Willemse, Kurt-Lee Arendse, Jesse Kriel, Damian de Allende, Cheslin Kolbe, Manie Libbok, Cobus Reinach; Steven Kitshoff, Bongi
Mbonambi, Frans Malherbe, Eben Etzebeth, Franco Mostert, Siya Kolisi, Pieter-Steph du Toit, Duane Vermeulen.

South Africa replacements: Deon Fourie, Ox Nche, Vincent Koch, RG Snyman, Kwagga Smith, Faf de Klerk, Handre Pollard, Willie le Roux.
</item 72>

<item 73>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Also Read: IND vs AUS World Cup 2023 Final: Weather Forecast, Pitch Report at Narendra Modi Stadium Ahmedabad ahead of India vs Australia clash

This is the fourth time that India has qualified for the ODI World Cup final, whereas for Australia, it is their eighth time. India faced Australia a total of
13 times in the ODI World Cups, with the Aussies dominating with eight wins. However, India has won the last two matches, most recently in the league fixture of
the 2023 World Cup in Chennai.
</item 73>

<item 74>
When was the first Cricket World Cup? History, winners' list and tournament details ahead of 2023 men's ODI showpiece (Sporting News, 2023-10-01T23:10:05+00:00)
The heart and soul of ICC Championships — the ODI Men's Cricket World Cup — will be held in India from October 5 to November 19.

Organised every four years by the International Cricket Council (ICC), the event is one of the most viewed sporting tournaments in the world.

Here we take a look at the history of the competition, and some of the landmark moments that have helped make the World Cup what it is today.

MORE: Who will win the World Cup? Make your prediction and bet here (selected territories)

When was the first ODI Cricket Word Cup?

England hosted the inaugural ODI World Cup back in 1975, and held the event the following three times too.

Initially, the game was played in cricket whites and with a red ball instead of a white ball.

Eight teams were a part of that inaugural tournament — India, Pakistan, England, Australia, West Indies, New Zealand and a composite team from East Africa and
Sri Lanka.
</item 74>

<item 75>
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
</item 75>

<item 76>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
2011 World Cup, against Pakistan

Fans from India and Pakistan wouldn't have expected a better occasion than a World Cup for the meeting of two teams.

The World Cup returned to India after 1996, and India and Pakistan were facing each other in the second semi-final at the IS Bindra Stadium in Mohali on March
30.

Man-of-the-Match Tendulkar rose to the occasion, scoring 85 runs for India.

Virender Sehwag scored 38 while Suresh Raina remained unbeaten on 36 as India scored 260 for 9 in 50 overs.

Pakistan seamer Wahab Riaz took 5 for 46 in 10 overs. In reply, Pakistan batsmen Misbah-ul-Haq (56 runs) and Mohammad Hafeez (43 runs) showed some resistance
from Pakistan, but in lack of a big innings lower down the order hit Pakistan's prospects badly as they were all out for 231 runs in 49.5 overs.
</item 76>

<item 77>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
TV channel Streaming USA — Peacock Canada TSN TSN+ UK ITV ITV X Australia Channel 9 Stan Sport, 9Now New Zealand Sky GO Sky GO India — Fan Code Hong Kong BeIN
Sports BeIN Sports Malaysia BeIN Sports BeIN Sports Singapore BeIN Sports BeIN Sports

MORE: Every single team to have lifted the Rugby World Cup trophy

England vs South Africa lineups, team news

"The start is crucial, the end is vital..."



- Steve Borthwick explains changing 3️⃣ of his England starters for Saturday's Rugby World Cup semi-final, from Liam Heagney 👨‍💻 in Paris. #RWC2023 #ENGvRSA
#EnglandRugby
</item 77>

<item 78>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
Also Read: IND vs NZ FREE Live Streaming: When and How to watch India vs New Zealand Cricket World Cup 2023 Semi-Final Match Live on Web, TV, mobile apps online

Also Read: AUS vs SA FREE Live Streaming: When and How to watch Australia vs South Africa Cricket World Cup 2023 Semi-Final Match Live on Web, TV, mobile apps
online

1983 Prudential Cup, against England

Having lost in the first round of the 1975 and 1979 World Cups, India were underdogs in the third World Cup, which was being held in England for the third
consecutive time.

However, in the World Cup, India wronged many of their critics, reaching semi-finals and playing against mighty hosts England in Manchester on June 22.

Skipper Kapil Dev struck three wickets while Roger Binny and Man-of-the-Match Mohinder Amarnath took two each as India restricted England to 213 all out in 60
overs.
</item 78>

<item 79>
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
</item 79>

<item 80>
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
</item 80>

<item 81>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
What a difference two days make. Somehow Tuesday’s ludicrous, record-smashing, high-scoring entertainment was followed by a sequel on the same surface that
eventually delivered thrilling low-scoring entertainment, and a different result claimed in very different style. It was eventually won for West Indies with
four balls to spare and in the most unfitting of ways – with a massive six – by Shai Hope, again proving himself the right man for a run-chase.

So a series marked by stunning power hitting, roaring run rates and an unexpected midway transformation in England’s performances and prospects concluded with a
3-2 triumph for West Indies, who extended their supremacy over England at home to four series in three formats across nearly five years.

England assembled exactly the same players in exactly the same place and exactly the same situation – losing the toss, put in to bat – but this was an entirely
different performance.
</item 81>

<item 82>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The 2019 Rugby World Cup final between England and South Africa was one of many firsts: the game itself capped off a sensational tournament played outside of
the sport's traditional strongholds for the first time; it was the first final in which South Africa scored a try, and by full-time, the 32-12 score in the
Springboks' favour sealed a record-equalling third title for the Rainbow Nation.

Fast forward four years, and South Africa once again prepare to take on England in an all-or-nothing match to decide which nation gets a shot at World Cup glory
— but this time around, the pair meet in the semi-finals of the 2023 tournament, rather than the decider.

That won't make an ounce of difference for either camp, though, as these two long-time rivals prepare for a battle that could re-define the definition of a
'physical encounter.'
</item 82>

<item 83>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
Here is the list of the top 10 batsmen and bowlers in this edition of the ODI World Cups—

Top 10 Batsmen of ODI World Cup 2023:

Rank Player Mat Runs 1 V Kohli (IND) 11 765 2 RG Sharma (IND) 11 597 3 Q de Kock (SA) 10 594 4 R Ravindra (NZ) 10 578 5 DJ Mitchell (NZ) 10 552 6 DA Warner
(AUS) 11 535 7 SS Iyer (IND) 11 530 8 KL Rahul (IND) 11 452 9 HE van der Dussen (SA) 10 448 10 MR Marsh (AUS) 10 441

*Still Batting

10.
</item 83>

<item 84>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
In short, this semi-final requires a huge step up in quality for Borthwick's side, and that's why we think that the Springboks will be victorious at the Stade
de France. South Africa have already shown they can match, and beat, the world's best, and we think that they'll do the latter in Saint-Denis.
</item 84>

<item 85>
England thrash Scotland but cannot avoid Team GB Olympic heartbreak (The Guardian, 2023-12-05T21:56:19+00:00)
The goal that gave England a cushion came shortly after the restart, Georgia Stanway beating Corsie to the byline before sending the ball into the middle for
Fran Kirby to turn in from close range. The breathing room didn’t last long, though, with the Netherlands scoring once more.

There was almost disaster for England when Earps failed to collect from the feet of Kirsty Hanson, but the Scotland forward sent her effort wide of the empty
net with the angle closing.

An added-time save of a header from the substitute Martha Thomas at full stretch by Mary Earps kept England in it. The Netherlands having scored in the 91st
minute, Bronze’s powering header meant England were back in for two minutes, until they were out, the Dutch scoring in the 95th minute to crush the Lionesses
while they stood in a huddle waiting for news of the other game’s result, the pain sweeping across their faces as it came through.
</item 85>

<item 86>
When was the first Cricket World Cup? History, winners' list and tournament details ahead of 2023 men's ODI showpiece (Sporting News, 2023-10-01T23:10:05+00:00)
India & Sri Lanka Sri Lanka Australia Sri Lanka won by seven wickets 1999 England Australia Pakistan Australia won by eight wickets 2003 South Africa Australia
India Australia won by 125 runs 2007 West Indies Australia Sri Lanka Australia won by 53 runs (DLS) 2011 India, Sri Lanka & Bangladesh India Sri Lanka India won
by six wickets 2015 Australia & New Zealand Australia New Zealand Australia won by seven wickets 2019 England & Wales England New Zealand England won on
boundary count after a Super Over

Who has won the most ODI Cricket World Cups?

Australia are the most successful nation in ODI World Cup history, with India and the West Indies chasing them.
</item 86>

<item 87>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Australia posted a target of 329 runs, losing seven wickets in the process. Steven Smith scored a brilliant 105 runs off 93 balls, with Aaron Finch supporting
the innings with an 81-run knock of 116 balls. Umesh Yadav picked up four wickets for India. India crumbled within 233 runs in 46.5 overs. MS Dhoni was the
highest scorer for the Indian side in that match, with 65 runs off 65 balls.

James Faulkner picked up three wickets for Australia, with Mitchell Johnson and Mitchell Stark taking two wickets each.

Australia went all the way to lift their fifth ODI World Cup trophy, defeating New Zealand in the final.

2019, Match 14, ICC Cricket World Cup, The Oval

India took their revenge in this World Cup as they annihilated the Australian bowlers by posting a target of 353 runs in 50 overs.
</item 87>

<item 88>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 23.36 GMT Jos Buttler’s reaction We probably didn’t adapt quickly enough with the bat – I thought 16o w0uld have been a decent score on a used wicket. It
was a fantastic bowling effort; we did well to hang in the game as long as we did. Phil Salt has had a brilliant series. We found out lots about ourselves as a
team. [On the stop-clock experiment] It was only in the last game that we really noticed it. It’s good to keep the game moving.



6d ago 23.19 GMT England’s miserable white-ball year ends with another series defeat. Plenty of good has come from this series, though, most notably the
performance of Phil Salt. While there are a few slots to fill, most notably at No3, they are in reasonable shape ahead of nexr year’s World Cup.
</item 88>

<item 89>
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
</item 89>

<item 90>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
ICC Men's ODI World Cup 2023 Most Runs and Wickets: With just the final of the ICC Men's ODI World Cup 2023 won by Australia beating India by six wickets on
Sunday, November 19, at Narendra Modi Stadium in Ahmedabad, the whole world is ready to witness who will be crowned the World Champions in the 50-over format of
the game. India defeated New Zealand by 70 runs in the first semi-final on Wednesday, November 15, at the Wankhede Stadium in Mumbai, and Australia beat South
Africa by three wickets in the second semi-final on Thursday, November 16, at the Eden Gardens in Kolkata.

During all these matches played, we witnessed some great knocks by the batter as well as some brilliant spells by the bowlers, which annihilated the opposition
and kept the audience in awe.
</item 90>

<item 91>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
22 3:00 a.m. SGT Malaysia Sun, Oct. 22 3:00 a.m. MYT Hong Kong Sun, Oct. 22 3:00 a.m. HKT

England vs South Africa live stream, TV channel

TV networks worldwide will be showing all 48 matches at the tournament live, including many on free-to-air television.

On social platforms alone, video content from the Rugby World Cup and organisers World Rugby drew more than 2.1 billion views in 2019 — almost six times the
previous figure.
</item 91>

<item 92>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
India set a target of 248 runs after being bowled out in 55.5 overs. Yashpal Sharma scored 40 off 40 for India in that innings. Jeff Thompson and Rodney Hogg
picked up three wickets each for Australia.

Australia crumbled within 129 runs, with Roger Binny and Madan Lal picking up four wickets for India. India went on to win their first-ever ODI World Cup,
defeating the West Indies in the final.

1987, Match 3, Reliance World Cup, Chennai

This was the first time the World Cup was hosted outside of England. This was one of the closest matches that have been played between these two sides, as
Australia won the match by just one run. After batting first, Australia posted a target of 271 runs courtesy of a 110-run off-141-ball knock by Geoff Marsh.
</item 92>

<item 93>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“We explain to the whole squad why we are going a certain route. It is not about individual selection,” Springboks coach Jacques Nienaber said shortly after
announcing two changes to his starting side and five in total, after going in with a forwards-heavy 7-1 bench split.

Advertisement

“It is for South Africa. As selectors we know there will be pressure on us if we don’t deliver a result but at the back of our minds we know this selection is
for South Africa.

“The players take it like that. It is not about person or ego, not that they have egos, but you can’t think of that. You have to think about South Africa and
the Springboks.”

Seemingly the Springboks are playing for more than themselves. They’re playing for their country.

It’s a powerful force, one that saw them progress through the knockout stages to reach the final after consecutive one-point victories.

Pressure?
</item 93>

<item 94>
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
</item 94>

<item 95>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
The All Blacks will experience a changing of the guard next season.

Indeed, the All Blacks’ social media channels described the team selection for the World Cup final as the “Last Dance”.

2015 World Cup winner Dane Coles will call time on his career, while his tight-five teammates Brodie Retallick and Sam Whitelock, who is bidding to become the
first player to win the Webb Ellis Cup for a third time, will head overseas.

Fellow World Cup winners Aaron Smith and Beauden Barrett will also head to Japan.

Coles, who was labelled by Samisoni Taukei’aho as a “Grandad” type figure in the All Blacks, has even been denied the chance of a farewell send-off after
missing selection.

Advertisement

“Yeah, tough,” said coach Ian Foster of the conversation with the 36-year-old around leaving him out.
</item 95>

<item 96>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
2. Rohit Sharma, India

The Indian captain has been on an all-out attack since the start of the World Cup campaign. He has smashed a total of 597 runs in 11 matches with one century
and three fifties so far.

1. Virat Kohli, India

One of the best batsmen that ever graced the sport of cricket, Virat Kohli, is at the top with 765 runs in 11 matches. He has five fifties and three centuries
to his name, with the highest score of 117 runs.
</item 96>

<item 97>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
ICC World Cup 2023: Rohit Sharma-led India are playing Netherlands in their ninth ICC World Cup 2023 round-robin match at the M Chinnaswamy Stadium in Bengaluru
on Sunday (November 12, 2023). India have already booked a place in semi-finals, thrashing Sri Lanka by 302 runs at the Wankhede Stadium in Mumbai on November
2, 2023. India will face New Zealand in the ongoing World Cup semi-final match at the Wankhede Stadium in Mumbai on Wednesday (November 15). This is the seventh
time India have entered into the last-four stage of the 50-over World Cup.

Also Read: ICC Cricket World Cup 2023 Prize Money: How much finalists, semi-finalists, and other teams will win at ODI World Cup

India first entered semifinals of the World Cup at the 1983 Prudential Cup in England, where they emerged as the tournament winners.
</item 97>

<item 98>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang

As the Premier League returns with a Saturday packed full of huge matches, talkSPORT gets the action started at Anfield.

But that's not all there is, with a Rugby World Cup semi-final, a London derby and a packed UFC card among the other offerings.

9 Merseyside derbies are never dull and talkSPORT has the first one of the season live Credit: Getty

9 Everton are starting to look much better after a rocky start Credit: Getty

Here, talkSPORT runs through what is an epic day of sport.

Merseyside derby

Liverpool are sitting in a comfortable fourth place in the Premier League table having picked up five wins out of their opening eight games.

The bad news for Jurgen Klopp is that the Reds lost to Tottenham and only managed a draw with Brighton just before the international break.
</item 98>

<item 99>
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
</item 99>

<item 100>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
In the episode, ‘ecky thump’ is a secret Lancastrian martial art using black puddings as a weapon. I’m not suggesting we go as far introducing black puddings
immediately. A phased withdrawal of the cricket bat will cause less fuss.” But what would you call The Hundred? Bloody Eck?

Updated at 22.00 GMT

6d ago 21.54 GMT WICKET! West Indies 20-1 (King c Buttler b Topley 3) One down, nine to go. Brandon King slogs Topley miles in the air towards mid-on, where
Buttler takes a simple catch. Jos Buttler takes the catch to dismiss Brandon King, and the first West Indies wicket is gone. Photograph: Randy Brooks/AFP/Getty
Images King traipses off the pitch.
</item 100>
=== END ITEM id=987594b3de559c35 ===
