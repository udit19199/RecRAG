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

=== ITEM id=0b3c27d46606979f ===
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
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
IND vs AUS Head-to-Head Record in ODI World Cup: It feels like a call back from the past, as two decades ago, these two goliaths—India and Australia—of the
cricketing world faced each other in the final of the ICC Men's ODI World Cup in Johannesburg, South Africa.

Also Read: India vs Australia World Cup Final Free Live Streaming: When and how to watch ICC World Cup 2023 Final IND vs AUS match live on Mobile apps, tv,
laptop, online

Once again, we are at the same stage as the two-time World Champions India will be locking horns against the five-time World Champions Australia for the finale
of the ICC Men's ODI World Cup 2023 on Sunday, November 19, at the Narendra Modi Stadium in Ahmedabad.
</item 1>

<item 2>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
India fought well, courtesy of two well-made 70s by Krishnamachari Srikkanth and Navjot Singh Sidhu. Unfortunately, India was bowled out in 49.5 overs with 269
runs on the board. Craig McDermott took four wickets for Australia.

1987, Match 15, Reliance World Cup, Delhi

This is the second time India has met Australia in this World Cup, with India winning this one by 56 runs. India batting first set a target of 290 runs courtesy
of four half-centuries by Sunil Gavaskar, Navjot Singh Sidhu, Dilip Vengsarkar, and Mohammad Azharuddin. Australia was bowled out within 233 runs, with David
Boon scoring 62 off 59 balls. Mohammad Azharuddin and Maninder Singh picked up three wickets for India.
</item 2>

<item 3>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
He and Franco Mostert pack down together in the second row, while hooker Bongi Mbonambi, utility Deon Fourie, veteran Duane Vermeulen and the devilish Kwagga
Smith will all be looking to back up superb quarter-final performances to take the Springboks to a second consecutive World Cup final.

South Africa starting XV: Damian Willemse, Kurt-Lee Arendse, Jesse Kriel, Damian de Allende, Cheslin Kolbe, Manie Libbok, Cobus Reinach; Steven Kitshoff, Bongi
Mbonambi, Frans Malherbe, Eben Etzebeth, Franco Mostert, Siya Kolisi, Pieter-Steph du Toit, Duane Vermeulen.

South Africa replacements: Deon Fourie, Ox Nche, Vincent Koch, RG Snyman, Kwagga Smith, Faf de Klerk, Handre Pollard, Willie le Roux.
</item 3>

<item 4>
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
</item 4>

<item 5>
When was the first Cricket World Cup? History, winners' list and tournament details ahead of 2023 men's ODI showpiece (Sporting News, 2023-10-01T23:10:05+00:00)
Both the host teams met in the semi-finals at Eden Gardens, Kolkata. India's poor performance sparked an unruly crowd to protest against the home team and Sri
Lanka was declared the winner by default, due to the halt in play.

Sri Lanka then went on to win the trophy that year, and they were back to try and defend it three years later in England, rather than the usual four-year cycle.

Following the 1996 event the ICC opted to try and revamp the schedule, with three years between events rather than four. But the move proved unpopular and 1999
was the only tournament to follow that pattern.

When was the last time India hosted the ODI Cricket World Cup?

The last time India welcomed the ODI Cricket World Cup was in 2011, when they were co-hosts with Sri Lanka and Bangladesh.

Pakistan were barred from hosting due to terrorist attacks on the Sri Lankan team when touring the country a couple of years prior to the World Cup.
</item 5>

<item 6>
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
</item 6>

<item 7>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
For India, Ravindra Jadeja picked up three wickets, along with Kuldeep Yadav and Jasprit Bumrah picking up two each.

India was in a terrible position at the start of the second innings as both of the openers got out for a duck, and India lost three wickets for just two runs.

However, two brilliant and sensible innings by KL Rahul and Virat Kohli, where they scored 97 runs off 115 balls and 85 runs off 116 balls, took India to shore
as India won the match by six wickets.

Also Read: 2023 ICC World Cup: Virat Kohli vs Australia; know the ODI batting records and stats

Catch the latest stock market updates here. For all other news related to business, politics, tech, sports, and auto, visit Zeebiz.com.
</item 7>

<item 8>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England replacements: Theo Dan, Ellis Genge, Kyle Sinckler, Ollie Chessum, Billy Vunipola, Danny Care, George Ford, Ollie Lawrence.

The Springboks are unchanged from that epic quarter-final with France, having all played their part as Jacques Nienaber's side hung on in the second half. The
contribution of backs Kurt-Lee Arendse, Damian de Allende and Cheslin Kolbe saw all three cross for first-half tries, with Kolbe's coming off the back of a
charged-down conversion from France's Thomas Ramos. Preventing those two points proved crucial in the end.

The key contribution, though, came from Eben Etzebeth, who bulldozed his way over to score the only five-pointer of the second half and wrestle the game back
into South African hands.
</item 8>

<item 9>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The 2019 Rugby World Cup final between England and South Africa was one of many firsts: the game itself capped off a sensational tournament played outside of
the sport's traditional strongholds for the first time; it was the first final in which South Africa scored a try, and by full-time, the 32-12 score in the
Springboks' favour sealed a record-equalling third title for the Rainbow Nation.

Fast forward four years, and South Africa once again prepare to take on England in an all-or-nothing match to decide which nation gets a shot at World Cup glory
— but this time around, the pair meet in the semi-finals of the 2023 tournament, rather than the decider.

That won't make an ounce of difference for either camp, though, as these two long-time rivals prepare for a battle that could re-define the definition of a
'physical encounter.'
</item 9>

<item 10>
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
</item 10>

<item 11>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The last time Cheslin Kolbe played England at a Rugby World Cup... 🇿🇦 🏆#RWC2023 | #ENGvRSA pic.twitter.com/Riw2uaSaKa — Rugby World Cup (@rugbyworldcup) October
19, 2023

Both sides' journeys have seen twists, turns, and bumps in the road, but it's led them both to within 80 minutes of a Rugby World Cup final. England will no
doubt be looking to echo the spirit of 2019, when they beat the All Blacks 19-7 to reach the final, but in South Africa, they face a side who have taken their
game to new heights just to reach the semis.

MORE: 2023 Rugby World Cup final: Details, kick-off time and every team's route

What time does England vs South Africa kick off?
</item 11>

<item 12>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
2011 World Cup, against Pakistan

Fans from India and Pakistan wouldn't have expected a better occasion than a World Cup for the meeting of two teams.

The World Cup returned to India after 1996, and India and Pakistan were facing each other in the second semi-final at the IS Bindra Stadium in Mohali on March
30.

Man-of-the-Match Tendulkar rose to the occasion, scoring 85 runs for India.

Virender Sehwag scored 38 while Suresh Raina remained unbeaten on 36 as India scored 260 for 9 in 50 overs.

Pakistan seamer Wahab Riaz took 5 for 46 in 10 overs. In reply, Pakistan batsmen Misbah-ul-Haq (56 runs) and Mohammad Hafeez (43 runs) showed some resistance
from Pakistan, but in lack of a big innings lower down the order hit Pakistan's prospects badly as they were all out for 231 runs in 49.5 overs.
</item 12>

<item 13>
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
</item 13>

<item 14>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
ICC Men's ODI World Cup 2023 Most Runs and Wickets: With just the final of the ICC Men's ODI World Cup 2023 won by Australia beating India by six wickets on
Sunday, November 19, at Narendra Modi Stadium in Ahmedabad, the whole world is ready to witness who will be crowned the World Champions in the 50-over format of
the game. India defeated New Zealand by 70 runs in the first semi-final on Wednesday, November 15, at the Wankhede Stadium in Mumbai, and Australia beat South
Africa by three wickets in the second semi-final on Thursday, November 16, at the Eden Gardens in Kolkata.

During all these matches played, we witnessed some great knocks by the batter as well as some brilliant spells by the bowlers, which annihilated the opposition
and kept the audience in awe.
</item 14>

<item 15>
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
</item 15>

<item 16>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India, one of the favourites to win the World Cup, were very much on the path to victory till Mohammad Azharuddin, who top-scored with 64 runs was at the
crease.

But with the fall of his wicket, the Indian lower order collapsed like a house of placards as India lost the match by 35 runs.

1996 Wills World Cup, against Sri Lanka

The 1996 Cricket World Cup is known for Sri Lanka's emergence as a cricketing power on the horizon of world cricket and the beginning of Sachin Tendulkar's
extraordinary dominance in batting history.

Playing in the first semi-final at the Eden Gardens in Kolkata on March 13, India were favourites again, and they raised their hopes when they managed to
restrict Sri Lanka to 251 for 8 in 50 overs despite Man-of-the-Match Aravinda de Silva's 66 and Roshan Mahanama's 58.
</item 16>

<item 17>
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
</item 17>

<item 18>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
ICC World Cup 2023: Rohit Sharma-led India are playing Netherlands in their ninth ICC World Cup 2023 round-robin match at the M Chinnaswamy Stadium in Bengaluru
on Sunday (November 12, 2023). India have already booked a place in semi-finals, thrashing Sri Lanka by 302 runs at the Wankhede Stadium in Mumbai on November
2, 2023. India will face New Zealand in the ongoing World Cup semi-final match at the Wankhede Stadium in Mumbai on Wednesday (November 15). This is the seventh
time India have entered into the last-four stage of the 50-over World Cup.

Also Read: ICC Cricket World Cup 2023 Prize Money: How much finalists, semi-finalists, and other teams will win at ODI World Cup

India first entered semifinals of the World Cup at the 1983 Prudential Cup in England, where they emerged as the tournament winners.
</item 18>

<item 19>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The latter is winning only his seventh England cap but gets the starting nod alongside Maro Itoje, with Genge, Chessum and the rest of the replacements hoping
to make an impact sizeable enough to overcome the Springboks' infamous 'bomb squad'.

Their time is now 🙌



Here is the @EnglandRugby team to take on reigning champions South Africa in the semi-final in Paris#RWC2023 | #ENGvRSA pic.twitter.com/nkf6rXBPUU — Rugby World
Cup (@rugbyworldcup) October 19, 2023

England starting XV: Freddie Steward, Jonny May, Joe Marchant, Manu Tuilagi, Elliot Daly, Owen Farrell, Alex Mitchell; Joe Marler, Jamie George, Dan Cole, Maro
Itoje, George Martin, Courtney Lawes, Tom Curry, Ben Earl.
</item 19>

<item 20>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India were playing their sixth semi-finals at a place where they played their first World Cup semi-final in 1983, Manchester. However, with a different team-
New Zealand.

Indian bowlers raised hopes when they restricted New Zealand to 239 for 8 in 50 overs with Bhuvneshwar Kumar taking three wickets.

Ross Taylor 74 runs (90 balls) and skipper Kane Willamson 67 runs (95 balls) were the top scorers for New Zealand. But Kiwi Man-of-the-Match bowler Matt Henry
sent openers KL Rahul and Rohit Sharma back to the pavilion soon, followed by the dismissal of Dinesh Karthik in the 10th over of India's innings.

With the Indian top-order failing, Ravindra Jadeja (77 runs in 59 balls) and wicketkeeper MS Dhoni (50 runs in 72 balls) tried to steady the Indian ship.

But once they fell, it was all over for India as they lost the battle by 18 runs.
</item 20>

<item 21>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
Click below for more ⤵️ https://t.co/LeUuw4kPfP — RugbyPass (@RugbyPass) October 19, 2023

Marcus Smith hardly put a foot wrong at fullback against Fiji, but the world champions are an entirely different proposition for England to deal with at the
back. The high-ball security and hulking frame of Freddie Steward returns to the No. 15 jersey in one of three changes, while at inside centre, Manu Tuilagi
starts hoping to repeat his heroics of four years ago, when his early try in a dominant semi-final performance took England past the All Blacks.

England's other two changes come in the tight five, with Steve Borthwick making some big calls ahead of this game. Joe Marler starts ahead of vice-captain Ellis
Genge at loosehead prop, and Ollie Chessum is replaced in the second-row by George Martin.
</item 21>

<item 22>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
TV channel Streaming USA — Peacock Canada TSN TSN+ UK ITV ITV X Australia Channel 9 Stan Sport, 9Now New Zealand Sky GO Sky GO India — Fan Code Hong Kong BeIN
Sports BeIN Sports Malaysia BeIN Sports BeIN Sports Singapore BeIN Sports BeIN Sports

MORE: Every single team to have lifted the Rugby World Cup trophy

England vs South Africa lineups, team news

"The start is crucial, the end is vital..."



- Steve Borthwick explains changing 3️⃣ of his England starters for Saturday's Rugby World Cup semi-final, from Liam Heagney 👨‍💻 in Paris. #RWC2023 #ENGvRSA
#EnglandRugby
</item 22>

<item 23>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The Springboks have done it tough, too, having already faced two of the tournament favourites on their way to the semi-finals. Comfortable defeats of Scotland,
Romania and Tonga couldn't mask the disappointment of the 13-8 loss to Ireland, which consigned South Africa to second in Pool B. However, with Ireland sent
home by the All Blacks in the quarters, the Bokke showed their World Cup mettle to edge the hosts 29-28 at the Stade de France, sneaking past Les Bleus in one
of the best games of rugby you're ever likely to see.
</item 23>

<item 24>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
Top 10 Bowlers of ODI World Cup 2023:

Rank Player Mat Wkts 1 Mohammed Shami (IND) 7 23 2 A Zampa (AUS) 11 23 3 D Madushanka (SL) 9 21 4 JJ Bumrah (IND) 11 20 5 G Coetzee (SA) 8 20 6 Shaheen Shah
Afridi (PAK) 9 18 7 M Jansen (SA) 9 17 8 RA Jadeja (IND) 11 16 9 Josh Hazlewood 11 16 10 MJ Santner (NZ) 10 16 11 BFW de Leede (NED) 9 16

*Still Bowling

10.
</item 24>

<item 25>
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
</item 25>

<item 26>
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
</item 26>

<item 27>
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
</item 27>

<item 28>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
2023 World Cup, against New Zealand

To be played
</item 28>

<item 29>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Shikhar Dhawan scored a blistering 117 runs off 109 balls, with Virat Kohli joining in with 82 runs off 77 balls. In the second innings, Indian bowlers bowled
out the Australian side within 316 runs, with Bhuvneshwar Kumar and Jasprit Bumrah picking up three wickets and Yuzvendra Chahal picking two as India won the
match by 36 runs.

2023, 5th Match, ICC Cricket World Cup, Chennai

The most recent clash between these two sides happened in this World Cup. Australia won the toss, chose to bat first, and posted a target of 200 runs as they
were bowled in 49.3 overs. David Warner scored 41 runs off 52 balls, and Steven Smith scored 46 runs off 71 balls in that innings.
</item 29>

<item 30>
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
</item 30>

<item 31>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The #Springboks are gearing up for another tough challenge this week with a semi-final clash against England 🇿🇦🏴󠁧󠁢󠁥󠁮󠁧󠁿#StrongerTogether #RWC2023
pic.twitter.com/HLSnnjWggB — Springboks (@Springboks) October 18, 2023

England are the only side still competing who are yet to lose a game, and that will stand them in good stead coming into this semi-final. Dominant wins over
Argentina, Japan and Chile sealed passage to the last eight, and after being given an almighty scare by Samoa in their final pool match, England overcame a
Fijian fightback in Marseille to triumph 30-24. Despite being outscored by three tries to two, captain Owen Farrell kicked 20 points, sending his side into the
last four at the expense of 2023's breakout stars.
</item 31>

<item 32>
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
</item 32>

<item 33>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India seamer Javagal Srinath took three top-order wickets, including Sanath Jayasuriya and Romesh Kaluwitharana.

Fans at the Eden Garden were cheering loudly till Tendulkar was at the crease.

But once he fell for 65 with the team at 98, India suffered one of the worst collapses in the batting history of the World Cups.

The crowd turned angry and hostile towards Sri Lanka, creating a ruckus at the stadium, and the match was interrupted.

With no respite from onlookers' side, umpires decided to call off the match when India were reeling at 120 for eight in 34.1 overs.

The long-lasting memories of the crowd's bad behaviour and a weeping Vinod Kambli returning to the pavilion are still fresh in the memories of the cricket fans
of that generation.
</item 33>

<item 34>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Mohammad Azharuddin scored a brilliant 93 runs off 102 balls for India.

1996, Match 19, Wills World Cup, Mumbai

This match can be considered one of the most exciting matches played between these two sides. Australia batted first and posted a target of 259 runs in 50
overs, courtesy of a brilliant 126 runs off 135 balls by Mark Waugh. Venkatesh Prasad and Venkatapathy Raju picked up two wickets each for India.

India batted well courtesy of Sachin Tendulkar, who made a brilliant 90 runs off 84 balls, along with Sanjay Manjrekar, who scored 62 runs off 91 balls.
However, India was bowled out within 242 runs in 48 overs. Damien Fleming took five wickets for Australia as Australia won the match by 16 runs.
</item 34>

<item 35>
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
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England and South Africa face off in the 2023 Rugby World Cup's second semi-final, played at the Stade de France in Saint-Denis, France. The game kicks off at 9
p.m. local time on Saturday, October 21, 2023.

Here are the kick-off times for this game around the world:

Location Date Time USA Sat, Oct. 21 3:00 p.m. ET Canada Sat, Oct. 21 3:00 p.m. ET Australia Sun, Oct. 22 6:00 a.m. AEDT UK Sat, Oct. 21 8:00 p.m. BST New
Zealand Sun, Oct. 22 8:00 a.m. NZDT India Sun, Oct. 22 12:30 a.m. IST Singapore Sun, Oct. 22 3:00 a.m. SGT Malaysia Sun, Oct.
</item 37>

<item 38>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
India set a target of 248 runs after being bowled out in 55.5 overs. Yashpal Sharma scored 40 off 40 for India in that innings. Jeff Thompson and Rodney Hogg
picked up three wickets each for Australia.

Australia crumbled within 129 runs, with Roger Binny and Madan Lal picking up four wickets for India. India went on to win their first-ever ODI World Cup,
defeating the West Indies in the final.

1987, Match 3, Reliance World Cup, Chennai

This was the first time the World Cup was hosted outside of England. This was one of the closest matches that have been played between these two sides, as
Australia won the match by just one run. After batting first, Australia posted a target of 271 runs courtesy of a 110-run off-141-ball knock by Geoff Marsh.
</item 38>

<item 39>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
They moved into the last-four stage on the next two occasions when the World Cup was held in India, in 1987 and 1996.

India were in semi-finals in 2003 in South Africa.

Eight years later, they were again in last four at the 2011 World Cup as hosts, and repeated their feat in 2015, 2019, and 2023.

It means that in their last six World Cups, India have been in the semi-finals on five occasions.

Also Read: ICC Cricket World Cup 2023 semi-final schedule: India to play New Zealand in Mumbai; South Africa to face Australia in Kolkata; know venue, squads

In this write-up, ZeeBiz will take you through the memory lane of India's semifinal journey in the biggest cricket extravaganza in the world.
</item 39>

<item 40>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
22 3:00 a.m. SGT Malaysia Sun, Oct. 22 3:00 a.m. MYT Hong Kong Sun, Oct. 22 3:00 a.m. HKT

England vs South Africa live stream, TV channel

TV networks worldwide will be showing all 48 matches at the tournament live, including many on free-to-air television.

On social platforms alone, video content from the Rugby World Cup and organisers World Rugby drew more than 2.1 billion views in 2019 — almost six times the
previous figure.
</item 40>

<item 41>
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
</item 41>

<item 42>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
In short, this semi-final requires a huge step up in quality for Borthwick's side, and that's why we think that the Springboks will be victorious at the Stade
de France. South Africa have already shown they can match, and beat, the world's best, and we think that they'll do the latter in Saint-Denis.
</item 42>

<item 43>
When was the first Cricket World Cup? History, winners' list and tournament details ahead of 2023 men's ODI showpiece (Sporting News, 2023-10-01T23:10:05+00:00)
Current cricketing powerhouse South Africa was banned at the time from international sport due to its policy of racial segregation under the Apartheid system,
and thus did not take part in the first ODI World Cup.

The West Indies won the first tournament and defended the title in the second World Cup in 1979, while India's first taste of glory came when they clinched the
trophy in 1983 by defeating the West Indies by 43 runs in the final.

ODI World Cup full list of winners and finals

Year Host Winner Runner-up Result 1975 England West Indies Australia West Indies won by 17 runs 1979 England West Indies England West Indies won by 92 runs 1983
England India West Indies India won by 43 runs 1987 India & Pakistan Australia England Australia won by seven runs 1992 Australia & New Zealand Pakistan England
Pakistan won by 22 runs 1996 Pakistan,
</item 43>

<item 44>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Mohammad Azharuddin and Maninder Singh picked up three wickets for India.

1992, Match 12, Benson & Hedges World Cup, Brisbane

The game in which the rain spoiled not just the match but India's winning chances as they lost by just one run as revised target. Australia set a target of 237
runs in 50 overs, losing nine wickets in the process. Dean Jones scored 90 runs off 108 balls for Australia. Kapil Dev and Manoj Prabhakar picked up three
wickets each for India. In the second innings, rain interrupted play after 16.2 overs. The revised target was recalculated to 236 off 47 overs. India needed 5
runs off 4 balls. However, India was bowled out due to two run-outs on the very last ball. Mohammad Azharuddin scored a brilliant 93 runs off 102 balls for
India.
</item 44>

<item 45>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Also Read: IND vs AUS World Cup 2023 Final: Weather Forecast, Pitch Report at Narendra Modi Stadium Ahmedabad ahead of India vs Australia clash

This is the fourth time that India has qualified for the ODI World Cup final, whereas for Australia, it is their eighth time. India faced Australia a total of
13 times in the ODI World Cups, with the Aussies dominating with eight wins. However, India has won the last two matches, most recently in the league fixture of
the 2023 World Cup in Chennai.
</item 45>

<item 46>
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
</item 46>

<item 47>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England vs. South Africa betting odds, prediction

The last time these two sides met was a tumultuous period to say the least, particularly for English rugby. South Africa's dominant 27-13 victory at Twickenham
proved to be the last game in charge for Eddie Jones, the coach who had taken England to the 2019 World Cup final. In his place, Steve Borthwick was hired, and
he's just one game away from repeating that feat.

England's form at this tournament has been perfect — at face value. Argentina, Japan, Chile, Samoa and Fiji have all provided highlights — Los Pumas are also in
the semi-finals — and England have beaten all of them. However, only Japan and Chile were defeated with bonus points, and the narrow wins over both Samoa and
Fiji in their last two games showed that England are certainly gettable.
</item 47>

<item 48>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
1. Mohammad Shami, India

Shami announced his return to the World Cup with a bang, picking a fifer in the very first match he played for India. He has picked up 24 wickets so far in just
six matches, conceding just 257 runs. He has picked up three five-wicket hauls including a seven-wicket haul against New Zealand in the semi-final on Wednesday,
November 15 at the Wankhede Stadium in Mumbai.

Catch the latest stock market updates here. For all other news related to business, politics, tech, sports, and auto, visit Zeebiz.com.
</item 48>

<item 49>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
Also Read: ICC Cricket World Cup 2023 Prize Money: How much finalists, semi-finalists, and other teams will win at ODI World Cup

Also Read: ICC ODI Cricket World Cup 2023, Winners List from 1975 to 2023: A look back to the history of the ODI World Cup

Plenty of wounds are still to be healed for India, as the defeat in the 2003 final against Australia still feels fresh. It is only time to tell if the Aussies
will repeat history or if it will be India who will write history by winning their third ODI World Cup.
</item 49>

<item 50>
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
</item 50>
=== END ITEM id=0b3c27d46606979f ===

=== ITEM id=19220af10e494215 ===
Gold sentences and facts:
Sentence 0: The feature is currently available in France, Germany, Japan, the U.K. and the U.S., with plans for further expansion.
  Fact 0.0: The feature is currently available in France.
  Fact 0.1: The feature is currently available in Germany.
  Fact 0.2: The feature is currently available in Japan.
  Fact 0.3: The feature is currently available in the U.K.
  Fact 0.4: The feature is currently available in the U.S.
  Fact 0.5: There are plans for further expansion of the feature.
Sentence 1: LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
  Fact 1.0: LinkedIn leans into AI tools.
  Fact 1.1: LinkedIn this week unveiled a string of new AI features.
  Fact 1.2: The new LinkedIn AI features span its job hunting products.
  Fact 1.3: The new LinkedIn AI features span its marketing products.
  Fact 1.4: The new LinkedIn AI features span its sales products.
  Fact 1.5: Ingrid writes about LinkedIn's new AI features.

Retrieved items:
<item 1>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
The Boston Dynamics AI Institute is still in its infancy, and therefore doesn’t have much in the way of public facing research to show just yet. It is, however,
doing a lot of hiring. This week it announced that MIT’s Kate Darling will lead research around “ethics and societal impact of robotics and AI.” From the
institute:

Darling’s team will explore immediate as well as long term questions on the implementation and use of robotics, impact on the workplace, infrastructure and
other topics. The team will perform studies and experiments designed to generate data needed for others to make informed ethics and policy decisions, and will
also develop a series of talks and workshops at the intersection of ethics, law, economics and robotics that will offer a platform for broad discussion.

“I’ve heard Gill say that making a car is taking small bits of metal and banging them up, welding them together and attaching them,” Raibert says. “That’s a
little on the primitive side. Now we have software, robotics, AI and all that stuff, and the car companies need to embrace that.
</item 1>

<item 2>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
This got the attention of another Silicon Valley powerhouse, Google, and specifically Page. He saw a demonstration of DeepMind’s machine playing Atari games. He
wanted in. The talent auction In the spring of 2012, Geoffrey Hinton, a 64-year-old professor at the University of Toronto, and two graduate students published
a research paper that showed the world what AI could do. They trained a neural network to recognise common objects such as flowers, dogs and cars.

Scientists were surprised by the accuracy of the technology built by Hinton and his students. One who took particular notice was Yu Kai, an AI researcher who
had met Hinton at a research conference and had recently started working for Baidu, a giant Chinese internet company. Baidu offered Hinton and his students
$US12 million to join the company in Beijing, according to three people familiar with the offer. Hinton turned Baidu down, but the money got his attention.
</item 2>

<item 3>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
“This is another indication that AI is progressing faster than even many experts anticipated,” Victoria Krakovna, who would soon join DeepMind as an “AI safety”
researcher, wrote in a blog post. DeepMind’s founders were increasingly worried about what Google would do with their inventions. In 2017, they tried to break
away from the company. Google responded by increasing the salaries and stock award packages of the DeepMind founders and their staff. They stayed put. The
ethics board never had a second meeting. The Breakup

Convinced that Page’s optimistic view of AI was dead wrong, and angry at his loss of DeepMind, Musk built his own lab. OpenAI was founded in late 2015, just a
few months after he met with Altman at the Rosewood hotel in Silicon Valley. Sam Altman Credit: Bloomberg Businessweek Musk pumped money into the lab, and his
former PayPal buddies – Hoffman and Thiel – came along for the ride.
</item 3>

<item 4>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Most read

AI, faked: Google unveiled a new flagship AI model this week called Gemini. But it didn’t release the full model, Gemini Ultra — only a “lite” version called
Gemini Pro. In a press briefing and blog posts, Google touted Gemini’s coding capabilities and multimodal prowess, claiming that the model can understand
images, audio and videos just as well as text. But Gemini Pro — which is strictly text-in, text-out — has proven to be mistake-prone. And in a worse look for
Google, the company was caught faking a Gemini demo by tuning text prompts with still images off camera.

Offensive notebooks: In another Google PR blunder, people who attended the company’s K&I Black Summit in August were given third-party notebooks containing
highly insensitive language. My colleague Dominic-Madori writes that the inside of the notebooks were printed with the phrase “I was just cotton the moment, but
I came back to take your notes” (emphasis ours).
</item 4>

<item 5>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
The EU’s voluntary anti-disinformation Code has 44 signatories in all — covering not just major social media and search platforms such as the aforementioned
giants but entities from across the ad industry and civil society organizations involved in fact-checking.

Google

On generative AI, Google’s report discusses “recent progress in large-scale AI models” which it suggests has “sparked additional discussion about the social
impacts of AI and raised concerns on topics such as misinformation”. The tech giant is an early adopter of generative AI in search — via its Bard chatbot.

“Google is committed to developing technology responsibly and has published AI Principles to guide our work, including application areas we will not pursue,” it
writes in summary on the topic, adding: “We have also established a governance team to put them into action by conducting ethical reviews of new systems,
avoiding bias and incorporating privacy, security and safety.

“Google Search has published guidance on AI-generated content, outlining its approach to maintaining a high standard of information quality and the overall
helpfulness of content on Search.
</item 5>

<item 6>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
When considering the landscape of foundational layer model providers like Alphabet/Google’s Bard, Microsoft/OpenAI’s GPT-4, and Anthropic’s Claude, it’s evident
that the more significant players possess inherent advantages regarding data accessibility, talent pool and computational resources. We expect this layer to
settle into an oligopolistic structure like the cloud provider market, albeit with the addition of a strong open source contingency that will drive considerable
third-party adoption.

As we look at the generative AI tech stack, the largest market opportunity lies above the model itself. Companies that introduce AI-powered APIs and operational
layers for specific industries will create brand-new use cases and transform workflows. By embracing this technology to revolutionize workflows, these companies
stand to unlock substantial value.

However, it’s essential to recognize that the market is still far from being crystallized. LLMs are still in their infancy, with adoption at large corporations
and startups lacking full maturity and refinement. We need robust tools and platforms to enable broader utilization among businesses and individuals.
</item 6>

<item 7>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
We need robust tools and platforms to enable broader utilization among businesses and individuals. Startups have the opportunity here to act quickly, find novel
solutions to emerging problems, and define new categories.

Interestingly, even large tech companies recognize the gaps in their services and have begun investing heavily in startups alongside VCs. These companies apply
AI to their internal processes and thus see the value startups bring to LLM deployment and integration. Consider the recent investments from Microsoft, Nvidia,
and Salesforce into companies like Inflection AI and Cohere.

What can be done to ensure industry-specific startups that tune generative AI models for a specific niche will prove defensible?

To ensure industry-specific startups will prove defensible in the rising climate of AI integration, startups must prioritize collecting proprietary data,
integrating a sophisticated application layer and assuring output accuracy.

We have established a framework to assess the defensibility of application layers of AI companies. First, the application must address a real enterprise pain
point prioritized by executives.
</item 7>

<item 8>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT: Everything you need to know about the AI-powered chatbot

ChatGPT, OpenAI’s text-generating AI chatbot, has taken the world by storm. What started as a tool to hyper-charge productivity through writing essays and code
with short text prompts has evolved into a behemoth used by more than 92% of Fortune 500 companies for more wide-ranging needs. And that growth has propelled
OpenAI itself into becoming one of the most-hyped companies in recent memory, even if CEO and co-founder Sam Altman’s firing and swift return raised concerns
about its direction and opened the door for competitors.

What does that mean for OpenAI, ChatGPT and its other ambitions? The fallout is still settling, but it might empower competitors like Meta and its LLaMA family
of large language models, or help other AI startups get attention and funding as the industry watches OpenAI implode and put itself back together.
</item 8>

<item 9>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
But costs are only one area where we have unanswered questions. We are also incredibly curious about how startups should approach building tools for AI
technologies, how defensible startup-focused AI work will prove, and how upstart tech companies should charge for AI-powered tooling.

With the amount of capital flowing to startups working with and building AI today, it’s critical that we understand the market as best we can. So we asked a
number of venture capitalists who are active in the AI investing space to walk us through what they are seeing in the market today.

What we learned from the investing side of the house was useful. Rick Grinnell, founder and managing partner at Glasswing Ventures, said that within the new AI
tech stack, “most of the opportunity lies in the application layer,” where “the best applications will harness their in-house expertise to build specialized
middle-layer tooling and blend them with the appropriate foundational models.”
</item 9>

<item 10>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
You cannot escape conversations about AI no matter how far or fast you run. Hyperbole abounds around what current AI tech will be able to do (revolutionize
every industry!) and what current AI tech will be able to do (take over the world!). Closer to the ground, TechCrunch+ is working to understand where startups
might find footholds in the market by levering large language models (LLMs), a recent and impactful new method of creating artificially intelligent software.

How AI will play in startup land is not a new topic of conversation. A few years back, one venture firm asked how AI-focused startups would monetize and whether
they would suffer from impaired margins due to costs relating to running models on behalf of customers. That conversation died down, only to come roaring back
in recent quarters as it became clear that while LLM technology is quickly advancing, it’s hardly cheap to run in its present form.

But costs are only one area where we have unanswered questions.
</item 10>

<item 11>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
But in the release, the AI Alliance stresses that its work — whatever form it ultimately takes — is intended to be complementary and additive rather than
needlessly duplicative.

“[M]ore collaboration and information sharing will help the community innovate faster and more inclusively, and identify specific risks and mitigate those risks
before putting a product into the world,” the release reads. “This stands in contrast to a vision that aims to relegate AI innovation and value creation to a
small number of companies with a closed, proprietary vision for the AI industry.”

Key subtext

That jab at the end says a lot about Meta’s ulterior motives, here.

Google, OpenAI and Microsoft, a close OpenAI partner and investor, have been among the chief critics of Meta’s open source AI approach, arguing that it’s
potentially dangerous and disinformation-encouraging. (Unsurprisingly, none are members of the AI Alliance despite being longtime members of the Partnership on
AI.)
</item 11>

<item 12>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
In our proprietary view of the GenAI tech stack, we categorize the landscape into four distinct layers: foundation model providers, middle-tier companies,
end-market or top-layer applications, and full stack or end-to-end vertical companies.

We think that most of the opportunity lies in the application layer, and within that layer, we believe that in the near future, the best applications will
harness their in-house expertise to build specialized middle-layer tooling and blend them with the appropriate foundational models. These are “vertically
integrated” or “full-stack” applications. For startups, this approach means a shorter time-to-market. Without negotiating or integrating with external entities,
startups can innovate, iterate and deploy solutions at an accelerated pace. This speed and agility can often be the differentiating factor in capturing market
share or meeting a critical market need before competitors.

On the other hand, we view the middle layer as a conduit, connecting the foundational aspects of AI with the refined specialized application layer.
</item 12>

<item 13>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Incumbents like Datadog, New Relic and Splunk have all produced LLM observability tools and do appear to be putting a lot of R&D dollars behind this, which may
curtail the market area in the short-term.

However, as we have seen before with the inceptions of the internet and cloud computing, incumbents tend to innovate until innovation becomes stagnant. With AI
becoming a household name that finds use cases in every vertical, startups have the chance to come in with innovative solutions that disrupt and reimagine the
work of incumbents. It’s still too early to say with certainty who the winners will be, as every day reveals new gaps in existing AI frameworks. Therein lie
major opportunities for startups.

How much room in the market do the largest tech companies’ services leave for smaller companies and startups tooling for LLM deployment?
</item 13>

<item 14>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
That’s effectively making robots that are smarter, better at mobile manipulation and move more dynamically. Raibert admits that “Boston Dynamics AI Institute”
might not be the most instructive name, both with regard to confusion between itself and Boston Dynamics the company, as well as a seeming focus on AI over
robotics. A name change is apparently coming, but nothing has been decided just yet.
</item 14>

<item 15>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Despite cloud computing remaining critical, AI budgets are now more than double that of cloud computing. Eighty-two percent of business leaders believe the
integration of AI solutions will increase their employee performance and job satisfaction, and startups should expect a high level of desire for and experience
with AI solutions in their future customers.

Finally, we’ve seen consumption, or usage-based priced tech products’ growth slow in recent quarters. Will that fact lead startups building modern AI tools to
pursue more traditional SaaS pricing? (The OpenAI pricing schema based on tokens and usage led us to this question.)

The trajectory of usage-based pricing has organically aligned with the needs of large language models, given that there is significant variation in
prompt/output sizes and resource utilization per user. OpenAI itself racks upward of $700,000 per day on compute, so to achieve profitability, these operation
costs need to be allocated effectively.

Nevertheless, we’ve seen the sentiment that tying all costs to volume is generally unpopular with end users, who prefer predictable systems that allow them to
budget more effectively.
</item 15>

<item 16>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
On the surface, Microsoft is currently looking like the biggest winner amidst the OpenAI turmoil, as it was already apparently looking to reduce its reliance on
OpenAI even though it remains once of its major shareholders. But Facebook’s parent Meta could also stand to benefit, as businesses pursue multi-modal
strategies or models with a more “open” ethos embedded.

“Open source today offers a wide variety of models for companies to essentially diversify,” Ceze added. “By doing so, these startups can quickly pivot and
minimize risk. There is also a major upside — many of these models already outperform the likes of OpenAI’s in terms [of] price-performance and speed.”

A leaked internal memo from Google earlier this year seemed to express fears that despite the huge advances made by proprietary LLM models from the likes of
OpenAI, open source AI would ultimately trump them all. “We have no moat, and neither does OpenAI,” the document noted.
</item 16>

<item 17>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
For $4.99, subscribers get an ad-free experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that
it’s piloting the plan in a single, English-speaking market outside the U.S. for now.

LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it throughout; an AI-powered LinkedIn Learning coach; and a
new AI-powered tool for marketing campaigns.

Musk comes clean about X’s metrics — maybe: In September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million
posts every day, excluding retweets. But speaking at an event this week, X CEO Linda Yaccarino offered a contradictory figure.
</item 17>

<item 18>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
Meta, on an open source tear, wants to spread its influence in the ongoing battle for AI mindshare.

This morning, the social network announced that it’s teaming up with IBM, whose audience is decidedly more corporate and enterprise, to launch the AI Alliance,
an industry body to support “open innovation” and “open science” in AI.

So what will the AI Alliance do exactly — and how will its work differ from the quite similar (at least in terms of its overarching mission, members and tenets)
Partnership on AI? The Partnership on AI years ago promised to publish research using open source licenses and minutes from its meetings to, as the AI Alliance
purportedly seeks to do, educate the public on pressing AI issues of the day.

Well — confusingly — the Partnership on AI is in fact a member of the AI Alliance.
</item 18>

<item 19>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
He gave Alan Eustace, Google’s senior vice president of engineering, what amounted to a blank check to hire any AI expertise he needed. Eustace and Jeff Dean,
who led the Brain lab, flew to Lake Tahoe and took Hinton and his students out to dinner at a steakhouse inside the hotel the night before the auction. The
smell of old cigarettes was overpowering, Dean recalled. They made the case for coming to work at Google. The next day, Hinton ran the auction from his hotel
room. Because of an old back injury, he rarely sat down. He turned a trash can upside down on a table, put his laptop on top and watched the bids roll in over
the next two days.

Google made an offer. So did Microsoft. DeepMind quickly bowed out as the price went up. The industry giants pushed the bids to $US20 million and then $US25
million, according to documents detailing the auction.
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
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
Now, those companies have a clear horse in the race and perhaps regulatory capture on the mind… but they’re not wrong entirely. Meta continues to take
calculated open sourcing risks (within the bounds of regulators’ tolerances), releasing text-generating models like Llama that bad actors have gone on to abuse
but which plenty of developers have built useful apps upon.

“The platform that will win will be the open one,” Yann LeCun, Meta’s chief AI scientist, was quoted as saying in an interview with The New York Times — and
who’s among the more than 70 influential signers of a letter calling for more openness in AI development. LeCun has a point; according to one estimate,
Stability AI’s open source AI-powered image generator, Stable Diffusion, released last August, is now responsible for 80% of all AI-generated imagery.

But wait, you might say — what does IBM gain from the AI Alliance? It’s a co-founder with Meta after all.
</item 21>

<item 22>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Marketing claims made for machine learning systems that companies adopted in order to not fall behind will receive their quarterly and yearly reviews… and it’s
very likely they will be found wanting.

Expect a considerable customer withdrawal from AI tools as the benefits fail to justify the costs and risks. On the far end of this spectrum, we are likely to
see lawsuits and regulatory action with AI service providers that failed to back up their claims.

While capabilities will continue to grow and advance, 2023’s products will not all survive by a long shot, and there will be a round of consolidation as the
wobblier riders of the wave fall and are consumed.

Apple jumps in

Apple has an established pattern of waiting, watching and learning from other companies’ failures, then blowing in with a refined and polished take that puts
others to shame. The timing is right for Apple to do this in AI, not just because if it waits too long its competition may eat up the market, but because the
tech is ripe for their kind of improvement.
</item 22>

<item 23>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hiya, folks, and welcome to Week in Review (WiR), TechCrunch’s digest of the past week in tech news. It’s TC’s column that highlights the major stories over the
past few days, and — we humbly submit — it’s a darn useful resource for folks on the go.

This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing
FTX fallout. Also on the agenda: Gmail’s harsher rules to prevent spam, TikTok testing an ad-free subscription plan, and LinkedIn going big on AI tools. And
that’s not all.

If you haven’t, sign up here to get WiR in your inbox every Saturday. And if you have, our thanks. Now, let’s get on with the news.
</item 23>

<item 24>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
Musk agreed. But it was pretty clear that his Google guests were not prepared to embark on a redistribution of (their) wealth. Schmidt said he thought the
worries were completely overblown. In his usual whisper, Page agreed. AI would create more jobs than it took away, he argued. Eight months later, DeepMind had a
breakthrough that stunned the AI community and the world. A DeepMind machine called AlphaGo beat one of the world’s best players at the ancient game of Go. The
game, streamed over the internet, was watched by 200 million people across the globe. Most researchers had assumed that AI needed another 10 years to muster the
ingenuity to do that. Demis Hassabis, right, co-founder of DeepMind, with South Korean professional Lee Se-dol. Lee played Go against the DeepMind machine
AlphaGo in March 2016. Credit: EPA

Rationalists, effective altruists and others who worried about the risks of AI claimed the computer’s win validated their fears.
</item 24>

<item 25>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
It’s a co-founder with Meta after all. I’d venture to guess more exposure for its burgeoning generative AI platform. IBM’s most recent earnings were boosted by
enterprises’ interest in generative AI, but the company has stiff competition in Microsoft and OpenAI (and to a lesser extent Google), which are jointly
developing enterprise-focused AI services that directly compete with IBM’s.

I’ve asked IBM’s PR, which first informed me of the AI Alliance’s founding, about the curious omissions from the early membership, like Stanford (which has a
prominent AI research lab, Stanford HAI), MIT (which is at the forefront of robotics research) and high-profile AI startups like Anthropic, Cohere and Adept. A
press rep didn’t respond as of publication time. But the same philosophical differences that kept Google and Microsoft away likely were at play; I’d wager it’s
no accident that Anthropic, Cohere and Adept have relatively few open source AI projects to their names.
</item 25>

<item 26>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
Well — confusingly — the Partnership on AI is in fact a member of the AI Alliance. The Alliance says that it plans to “utilize pre-existing collaborations”
(including the Partnership on AI’s, presumably) to “identify opportunities that develop open AI resources that meet the needs of business and society equally
and responsibly,” a press release shared last week with TechCrunch reads.

The AI Alliance’s members will first form working groups, a governing board and a technical oversight committee dedicated to advancing areas like AI “trust and
validation” metrics, hardware and infrastructure that supports AI training and open source AI models and frameworks. They’ll also establish project standards
and guidelines, and then partner with “important existing initiatives” — initiatives conspicuously not named in the press release — from government, nonprofit
and civil society organizations “who are doing valuable and aligned work in the AI space.”

If that sounds a lot like what the inaugural members of the Alliance were already doing independently, you’re not wrong.
</item 26>

<item 27>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
When Altman and others pushed back, Musk quit and said he would focus on his own AI work at Tesla. In February 2018, he announced his departure to OpenAI’s
staff on the top floor of the startup’s offices in a converted truck factory, three people who attended the meeting said. When he said that OpenAI needed to
move faster, one researcher retorted at the meeting that Musk was being reckless. Musk called the researcher a “jackass” and stormed out, taking his deep
pockets with him. OpenAI suddenly needed new financing in a hurry. Altman flew to Sun Valley for a conference and ran into Satya Nadella, Microsoft’s CEO. A
tie-up seemed natural. Altman knew Microsoft’s chief technology officer, Kevin Scott. Microsoft had bought LinkedIn from Hoffman, an OpenAI board member.
Nadella told Scott to get it done. The deal closed in 2019.
</item 27>

<item 28>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 28>

<item 29>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
By the end of 2012, Google and Facebook were angling to acquire the London lab, according to three people familiar with the matter. Hassabis and his co-founders
insisted on two conditions: No DeepMind technology could be used for military purposes, and its AGI technology must be overseen by an independent board of
technologists and ethicists.

Google offered $US650 million. Zuckerberg of Facebook offered a bigger payout to DeepMind’s founders but would not agree to the conditions. DeepMind sold to
Google. Zuckerberg was determined to build an AI lab of his own. He hired Yann LeCun, a French computer scientist who had also done pioneering AI research, to
run it. A year after Hinton’s auction, Zuckerberg and LeCun flew to Lake Tahoe for the same AI conference. While padding around a suite at the Harrah’s casino
in his socks, Zuckerberg personally interviewed top researchers, who were soon offered millions of dollars in salary and stock. AI was once laughed off.
</item 29>

<item 30>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
It is aiming to help any software play nice with ChatGPT and other LLM ecosystems.

What big eyes you have: Adobe continues to push for easier image editing, showing off its Project Stardust as a sneak preview of its next-gen AI photo editing
engine.

Holding back the tide: Creatives across industries are strategizing in a campaign that targets potential corporate abuse of AI technology. The conglomeration is
realistic about the ways that musicians and some other creatives could benefit on an individual level from automating parts of their work. The goal is that AI
tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

Tech you can touch

A while back I argued that Apple’s new AR headset is a game-changer for startups. It seems that’s likely the case at the high end. But more in the realm of
affordability, Meta Quest 3 takes a step closer to mainstream AR/VR, Brian reports.
</item 30>

<item 31>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
Hinton turned Baidu down, but the money got his attention. The Cambridge-educated British expatriate had spent most of his career in academia, except for
occasional stints at Microsoft and Google, and was not especially driven by money. But he had a neurodivergent child, and the money would mean financial
security. “We did not know how much we were worth,” Hinton said. He consulted lawyers and experts on acquisitions and came up with a plan: “We would organise an
auction, and we would sell ourselves.” The auction would take place during an annual AI conference at the Harrah’s hotel and casino on Lake Tahoe. Big Tech took
notice.

Google, Microsoft, Baidu and other companies were beginning to believe that neural networks were a path to machines that could not only see but also hear,
write, talk and — eventually — think. Loading Page had seen similar technology at Google Brain, his company’s AI lab, and he thought Hinton’s research could
elevate his scientists’ work.
</item 31>

<item 32>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
Nadella told Scott to get it done. The deal closed in 2019. Altman and OpenAI had formed a for-profit company under the original nonprofit, they had $US1
billion in fresh capital, and Microsoft had a new way to build AI into its vast cloud computing service. Not everyone inside OpenAI was happy.

Dario Amodei, a researcher with ties to the effective altruist community, had been on hand at the Rosewood hotel when OpenAI was born. Amodei, who endlessly
twisted his curls between his fingers as he talked, was leading the lab’s efforts to build a neural network called a large language model that could learn from
enormous amounts of digital text. Seeking the path to artificial general intelligence, AGI. Credit: iStock By analysing countless Wikipedia articles, digital
books and message boards, it could generate text on its own. It also had the unfortunate habit of making things up. It was called GPT-3, and it was released in
the summer of 2020.
</item 32>

<item 33>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
(Spoiler alert: Yes.)

Rumors are swilling that OpenAI may be considering developing its own AI chips. That’s going to get interesting, especially if you’ll recall that Nvidia’s
ongoing stock market rally is likely driven by the current boom in AI. The company’s ChatGPT’s mobile app hit a record $4.6 million in revenue last month, but
growth is slowing. Oh, and there’s no need to shed tears for Nvidia quite yet; Brian’s piece breaking down how Nvidia became a major player in robotics is super
interesting.

Adobe has doubled down on its Firefly generative AI models. This week, Frederic covered how the software can now generate more realistic images and can help
artists create vector graphics in Illustrator. Neato.

More on the AI front

If AI can’t go to the mountain, the mountain will come to AI: Dutch startup Tidalflow exits stealth with backing from Google’s Gradient Ventures. It is aiming
to help any software play nice with ChatGPT and other LLM ecosystems.
</item 33>

<item 34>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Within the enterprise sector, there’s a clear recognition of the value of AI. However, many lack the internal capabilities to develop AI solutions. This gap
presents a significant opportunity for startups specializing in AI to engage with enterprise clients. As the business landscape matures, proficiency in
leveraging AI is becoming a strategic imperative.

McKinsey reports that generative AI alone can add up to $4.4 trillion in value across industries through writing code, analyzing consumer trends, personalizing
customer service, improving operating efficiencies, and more. Ninety-four percent of business leaders agree AI will be critical to all businesses’ success over
the next five years, and total global spending on AI is expected to reach $154 billion by the end of this year, a 27% increase from 2022. The next three years
are also expected to see a compound annual growth rate of 27% — the annual AI spending in 2026 will be over $300 billion.
</item 34>

<item 35>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
The idea, O’Hara said, is to rethink social networking from the ground up.

Ford shutters SaaS app for field work: Ford has shut down VIIZR, a software-as-a-service company that, along with Salesforce, built an app to help tradespeople
like plumbers, locksmiths and electricians to schedule field appointments, send invoices and manage customers, Kirsten reports. VIIZR, which was announced in
December 2021, was a separate company majority owned by Ford, with Salesforce as a minority investor.

Apple bets on generative AI: Apple CEO Tim Cook pushed back against the notion that the company was behind in AI on Apple’s Q4 earnings call with investors, as
he highlighted technology developments that Apple had made recently that “would not be possible without AI.” Cook also said that Apple was working on generative
AI technologies, lending credence to reports suggesting the company is on track to spend $1 billion per year on developing generative AI products.
</item 35>

<item 36>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Hey, folks, welcome to Week in Review (WiR), TechCrunch’s regular newsletter that recaps the past few days in tech. AI stole the headlines once again, with tech
giants from Google to X (formerly Twitter) heading off against OpenAI for chatbot supremacy. But plenty happened besides.

In this edition of WiR, we cover Google faking a demo of its new AI model (and giving out offensive notebooks to Black summit attendees), defense startup
Anduril unveiling a fighter jet weapon, the continued fallout from the 23andMe hack, and the release of the Grand Theft Auto VI trailer. Also on the roster are
stories about patient scans and health records spilling online, Meta’s new AI-powered image generator, Spotify cutting jobs and an autonomous truck startup
leaving the U.S.

It’s a lot to get to, so we won’t delay. But first, a reminder to sign up here to receive WiR in your inbox every Saturday if you haven’t already done so.
</item 36>

<item 37>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
The latest funding round brings QI Tech’s total capital raised to $262 million.

New Web Summit CEO appointed: Former Wikimedia Foundation CEO Katherine Maher has been appointed as the new CEO of Web Summit, following the controversial
departure of its previous leader, Paddy Cosgrave. Maher’s appointment comes as the tech conference faces a challenge to rebuild relationships and trust within
the tech community.

Top reads on TechCrunch this week

Shinier, smarter maps: Google is rolling out a variety of updates for Maps, incorporating AI technology to enhance user experience. The feature is currently
available in France, Germany, Japan, the U.K. and the U.S., with plans for further expansion.

From stop-and-start to charging ahead: Over on TC+, Tim has had it with U.S. automakers hesitating on EVs, as Ford and GM are showing signs of backing away from
their commitment to electric vehicles. Cowards.
</item 37>

<item 38>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
First, the application must address a real enterprise pain point prioritized by executives. Second, to provide tangible benefits and long-term differentiation,
the application should be composed of cutting-edge models that fit the specific and unique needs of the software. It’s not enough to simply plug into OpenAI;
rather, applications should choose their models intentionally while balancing cost, compute, and performance.

Third, the application is only as sophisticated as the data that it is fed. Proprietary data is necessary for specific and relevant insights and to ensure
others cannot replicate the final product. To this end, in-house middle-layer capabilities provide a competitive edge while harnessing the power of foundational
models. Finally, due to the inevitable margin of error of generative AI, the niche market must tolerate imprecision, which is inherently found in subjective and
ambiguous content, like sales or marketing.

How much technical competence can startups presume that their future enterprise AI customers will have in-house, and how much does that presumed expertise guide
startup product selection and go-to-market motion?
</item 38>

<item 39>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
When Google bought DeepMind, it bought the whole thing. Musk was out. Financially, he had come out ahead, but he was unhappy. Three Google executives now firmly
in control of DeepMind were there: Page; Sergey Brin, a Google co-founder and Tesla investor; and Eric Schmidt, Google’s chair. Among the other attendees were
Reid Hoffman, another PayPal founder; and Toby Ord, an Australian philosopher studying “existential risk”.

The DeepMind founders reported that they were pushing ahead with their work but that they were aware the technology carried serious risks. Suleyman, the
DeepMind co-founder, gave a presentation called “The Pitchforkers Are Coming.” AI could lead to an explosion in disinformation, he told the board. He fretted
that as the technology replaced countless jobs in the coming years, the public would accuse Google of stealing their livelihoods. Google would need to share its
wealth with the millions who could no longer find work and provide a “universal basic income,” he argued. Musk agreed.
</item 39>

<item 40>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
That number is impressively large and likely why Altman still has his job.

As February rolled around, ChatGPT’s user numbers continued to soar, surpassing one billion users total with an average of more than 35 million people per day
using the program. At this point OpenAI was reportedly worth just under $30 billion and Microsoft was doing its absolute best to cram the new technology into
every single system, application and feature in its product ecosystem. ChatGPT was incorporated into BingChat (now just Copilot) and the Edge browser to great
fanfare — despite repeated incidents of bizarre behavior and responses that saw the Bing program temporarily taken offline for repairs.

Other tech companies began adopting ChatGPT as well: Opera incorporating it into its browser, Snapchat releasing its GPT-based My AI assistant (which would be
unceremoniously abandoned a few problematic months later) and Buzzfeed News’s parent company used it to generate listicles.
</item 40>

<item 41>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
The “app store for AI” will be pushed hard as the platform to get your AI toys and tools from, and never mind Hugging Face or any open source models. They have
an excellent model to work from, Apple’s, and will follow it all the way to the bank.

Expect more moves like that from 2024’s OpenAI as the caution and academic reserve that the previous board exerted gives way to an unseemly lust for markets and
customers.

Other major companies with AI efforts will also follow this trend (for instance, expect Gemini/Bard to horn in on a ton of Google products), but I suspect it
will be more pronounced in this case.

Agents, generated video and generated music graduate from quaint to experimental

Some niche applications of AI models will grow beyond “eh” status in 2024, including agent-based models and generative multimedia.

If AI is going to help you do more than summarize or make lists of things, it’ll need access to things like your spreadsheets, ticket buying interfaces,
transportation apps and so on.
</item 41>

<item 42>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
In addition to academic partners such as the University of Tokyo, UC Berkeley, the University of Illinois, Cornell and the aforementioned Imperial College
London and Yale, Sony, ServiceNow, the National Science Foundation, NASA, Oracle, the Cleveland Clinic and Dell have pledged their participation in some form.

MLCommons, the engineering consortium behind MLPerf, the benchmarking suite used by major chip manufacturers to evaluate their hardware’s AI performance, is
also a founding AI Alliance member. So are LangChain and LlamaIndex, two creators behind some of the more widely used tools and frameworks for building apps
powered by text-generating AI models.

But without the participation of so many major AI industry players — and lacking deadlines or even concrete objectives — can the AI Alliance succeed? What would
success look like, even?

Beats me.

The vast number of competing interests — from healthcare networks (Cleveland Clinic) to insurance providers (Roadzen) — won’t make it easy for the Alliance’s
members to coalesce around a single, united front.
</item 42>

<item 43>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
8 predictions for AI in 2024 How will AI impact the US primary elections? What's next for OpenAI? Here are our predictions for AI in 2024.

This last year was a banger for AI as the technology went from niche to mainstream about as fast as anything ever has. 2024, however, will be the year when the
hype runs full-steam into reality as people reckon with the capabilities and limitations of AI at large. Here are a few ways we think that’s going to play out.

OpenAI becomes a product company

After the leadership shake-up in November, OpenAI is going to be a changed company — perhaps not outwardly, but the trickle-down effect of Sam Altman being more
fully in charge will be felt at every level. And one of the ways we expect that to manifest is in “ship it” mindset.

We’ll see that with the GPT store, originally planned for launch in December but understandably delayed due to the C-suite fracas.
</item 43>

<item 44>
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
</item 44>

<item 45>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Do you expect startups to build striated solutions for individual layers of the LLM stack, or pursue a more vertical approach?

While there are startups specializing in parts of the stack (like Pinecone), Valor’s focus is on applied AI, which we define as AI that is solving a customer
problem. Saile.ai is a good example — it uses AI to generate closeable leads for the Fortune 500. Or Funding U using its own trained dataset to create a more
useful credit risk score. Or Allelica, using AI on treatment solutions applied to individual DNA to find the best medical treatment for you personally in a
given situation.

Companies like Datadog are building products to support the expanding AI market, including releasing an LLM observability tool. Will efforts like what Datadog
has built (and similar output from large/incumbent tech powers) curtail the market area where startups can build and compete?

Tools like Datadog can only help the acceptance of AI tools if they succeed in monitoring AI performance bottlenecks.
</item 45>

<item 46>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
I think the leadership at Hyundai — the chairman visited me last week, and we’re in close touch with him and the people who work with him — think that getting
into the 22nd century is an important thing to do and we’re getting started on it now.”

Pratt adds, “There’s actually this alignment that’s happening between cars and robotics, as well. Rod Brooks was the one who a few years ago said that modern
cars are eldercare robots. I think that’s really true. You think about the amount of computers in them, the amount of software that’s in them. TRI has different
divisions inside of it, one of which works on very advanced things in cars that is completely overlapping with some of the software and some of the concepts
that we’re using in the robotics space.”

For its part, the Boston Dynamics AI Institute lists three key pillars of its research: intelligence, dexterity and mobility. That’s effectively making robots
that are smarter, better at mobile manipulation and move more dynamically.
</item 46>

<item 47>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Battery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software that can be slipped into just about any
lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.

What lies beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched
on where startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.
</item 47>

<item 48>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
In a post on X, CEO Sam Altman looked back on the night before its launch: “what a year it’s been…”

a year ago tonight we were probably just sitting around the office putting the finishing touches on chatgpt before the next morning’s launch. what a year it’s
been… — Sam Altman (@sama) November 30, 2023

Apple and Google avoid naming ChatGPT as their ‘app of the year’

Neither Apple nor Google chose an AI app as its app of the year for 2023, despite the success of ChatGPT’s mobile app, which became the fastest-growing consumer
application in history before the record was broken by Meta’s Threads.

An attack from researchers prompts ChatGPT to reveal training data

A test led by researchers at Google DeepMind found that there is a significant amount of privately identifiable information in OpenAI’s LLMs.
</item 48>

<item 49>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
This part of the stack includes cutting-edge capabilities, encompassing model fine-tuning, prompt engineering and agile model orchestration. It’s here that we
anticipate the rise of entities akin to Databricks. Yet, the competitive dynamics of this layer present a unique challenge. Primarily, the emergence of
foundation model providers expanding into middle-layer tools heightens commoditization risks. Additionally, established market leaders venturing into this space
further intensify the competition. Consequently, despite a surge in startups within this domain, clear winners still need to be discovered.

Companies like Datadog are building products to support the expanding AI market, including releasing an LLM observability tool. Will efforts like what Datadog
has built (and similar output from large/incumbent tech powers) curtail the market area where startups can build and compete?

LLM observability falls within the “middle layer” category, acting as a catalyst for specialized business applications to use foundational models.
</item 49>

<item 50>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 50>
=== END ITEM id=19220af10e494215 ===
