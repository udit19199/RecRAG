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

=== ITEM id=c5b8fa9676990658 ===
Gold sentences and facts:
Sentence 0: The Springboks took a 12-3 lead after 34 minutes when the All Blacks once again failed to release on their goal line.
  Fact 0.0: The Springboks took a 12-3 lead after 34 minutes.
  Fact 0.1: The Springboks took the lead when the All Blacks once again failed to release on their goal line.
  Fact 0.2: The All Blacks failed to release on their goal line once again.
Sentence 1: Sensing an opportunity to strike against 14 men, the All Blacks kicked for the corner on a couple occasions out wide.
  Fact 1.0: The All Blacks sensed an opportunity to strike against 14 men.
  Fact 1.1: On a couple of occasions, the All Blacks kicked for the corner out wide.
Sentence 2: It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.
  Fact 2.0: This comes despite Argentina beating the All Blacks last year in Christchurch.
  Fact 2.1: Argentina also gave the All Blacks a maiden defeat in 2020 in Sydney.

Retrieved items:
<item 1>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
Against Sri Lanka, they were once again exposed.

“Too many of our guys just want to play crash, bang wallop all the time irrespective of the type of pitch or opposition. Again not smart.”

The brunt of Boycott’s criticism of England’s team composition was directed at the continued struggles of Jonny Bairstow, who has averaged just 25.2 this
tournament, and all-rounder Chris Woakes, whose nightmare campaign continued against Sri Lanka with figures of 0/30 from six overs and a fourth-ball duck.

“Bairstow has struggled since his horrific injury [breaking his leg 12 months ago],” Boycott wrote.

“I have said before I believe Jonny came back too soon and was short of fitness and match practice. In the recent five Ashes Tests his keeping at Lord’s was
poor with missed chances and he made only three significant batting scores out of nine innings.
</item 1>

<item 2>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
We’re not at that level. And we can’t pretend to be at that level. But can we be at that level by 2027? Yes, we can,” said Jones.

“We’re definitely off the pace but the opportunity to get better is there. I love the game. I love working with young players. I loved seeing some of the young
players maturing during the World Cup and some of them struggle. And that’s all part of it. But I love the game and I love coaching the Wallabies.”

Jones was named Wallabies coach at the start of the year as replacement for Dave Rennie. He came in with grand statements that his team could win the Rugby
Championship, Bledisloe Cup and make a smash and grab on the World Cup.
</item 2>

<item 3>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
But the first real in-match statement came on 27 minutes as New Zealand-born Aki – perhaps the player of the World Cup so far who had made a tournament-high 61
carries, 33 dominant carries and 23 defenders beaten heading into the weekend – brilliantly jinked inside two defenders, fended off another and powered through
a gap to dive over the line.

Yet more All-Black class saw Ardie Savea go over in the corner after quick ball was shipped wide but with scrum-half Aaron Smith in the sin-bin for a deliberate
knock-on, Ireland narrowed the gap to 18-17 just before the break as another New Zealand-born Irish star, Jamison Gibson-Park, sniped from a lineout maul and
somehow stretched out for the try.

The All Blacks extended an incredible record of leading at half-time of their quarter-final in every single World Cup but that lead had never felt more fragile.
</item 3>

<item 4>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
Click below for more ⤵️ https://t.co/LeUuw4kPfP — RugbyPass (@RugbyPass) October 19, 2023

Marcus Smith hardly put a foot wrong at fullback against Fiji, but the world champions are an entirely different proposition for England to deal with at the
back. The high-ball security and hulking frame of Freddie Steward returns to the No. 15 jersey in one of three changes, while at inside centre, Manu Tuilagi
starts hoping to repeat his heroics of four years ago, when his early try in a dominant semi-final performance took England past the All Blacks.

England's other two changes come in the tight five, with Steve Borthwick making some big calls ahead of this game. Joe Marler starts ahead of vice-captain Ellis
Genge at loosehead prop, and Ollie Chessum is replaced in the second-row by George Martin.
</item 4>

<item 5>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
“This group is very proud to be All Blacks, very proud to play for their country and have worked incredibly hard to get here.

“While I am so proud of what we did and how we worked we have to give it to South Africa, they are a quality team. They have had a few close victories in this
tournament and are a different class. It’s special for them but it equally hurts for us. I can’t ask for much more than we did, we couldn’t put more out on the
park.”

Advertisement

Controversy was on display from the outset, as Frizell was shown a yellow card for an illegal cleanout on Springboks hooker Bongi Mbonambi. The incident saw the
hooker, who was cleared earlier in the week by World Rugby following an alleged racial slur on England back-rower Tom Curry, replaced by Deon Fourie in the
third minute.

The penalty allowed Pollard to step up and bang over three points.
</item 5>

<item 6>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Horne’s close relationship with former Ireland coach and All Blacks assistant Joe Schmidt could be one of the keys to landing the 2018 World Rugby coach of the
year.

After leaving his Irish post after six years following the 2019 World Cup, Schmidt worked alongside Horne at World Rugby.

It’s believed the duo have a strong working relationship.

While Schmidt’s Irish side bombed out in successive quarter-finals under his leadership, the New Zealander took the Emerald Isle to the top of World Rugby’s
rankings.

Along the way, Schmidt played an instrumental role in not just seeing them become one of the world’s best teams but helped usher through transformational change
off the field, as Ireland moved to a centralised system.

Advertisement

Schmidt briefly exited the international coaching scene following the 2019 World Cup, citing a desire to move home to New Zealand to be closer to his family.
</item 6>

<item 7>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
“I was able to put lessons from that time into practice, develop and eventually return to the Socceroos much more prepared for the role. It’s been a privilege
to be part of the national team again with the benefit of that experience, and I’m immensely proud of what this group has been able to achieve in the last four
years.

“In my opinion, there’s no greater honour than representing your nation on the international stage. I look at the young players coming through the Socceroos
system now full of pride in the work they’ve put in to get where they are, and the journey that’s ahead of them.

“I’ve always loved being a part of the Socceroos culture and my desire to see Australia fulfil its footballing potential is still driving me to take this group
to further success.”

Australia starts the road to north America in Melbourne at AAMI Park on Thursday night, before travelling to the Middle East to play Palestine next Wednesday
(1.00am AEDT).
</item 7>

<item 8>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
You could feel the desperation around the Stade de France, awash with green thanks to the Irish fans doing their part, as Ireland went through phase after phase
after phase after phase hunting the try that could overturn the 28-24 scoreline staring back at them from the giant screen.

A move that started well inside their own half, with 77 minutes on the clock, ended 37 phases and five minutes later in heartbreak. Thirty-seven gruelling
phases of Bundee Aki grinding out extra yards with each punishing carry, Tadhg Beirne left at a literal standstill through tiredness and Johnny Sexton – for the
final time ahead of retirement – running the famous wraparound upon which he’s built maybe the greatest Ireland career of all-time. But it ended, almost
inevitably, with a turnover deep in the New Zealand 22, the ball being kicked to touch and the men in green sinking to the floor bereft.
</item 8>

<item 9>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
Wallabies coach Eddie Jones has again denied any contact with Japan and committed to staying in the job through to the next World Cup.

“That’s the intention, mate, but as you know, we play in a game where the coach doesn’t decide how long they stay,” he said at a media conference on Tuesday,
adding he had a “100 percent” passion to continue coaching the Wallabies.

“I’m staying. I’ve always been committed to Australian rugby, want to leave it in a better place and that’s still the job. “

Facebook Twitter Whatsapp Reddit Email Share

He was asked about his pre-tournament press conference where he attacked reporters’ negativity – “I probably need to give myself an uppercut, don’t I?” he
joked.

Jones said he accepted that a review would be required into the World Cup flop, but his statement puts the ball firmly in Rugby Australia’s court as far as his
future goes.
</item 9>

<item 10>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
I’ve got the experience of being in difficult situations before,” Jones said.

“This is not an unusual situation. You’ve got a team that was struggling for a period of time, you’ve got underneath it a system that’s not supporting it and at
some stage you bottom out.

“This is the opportunity now to change the team- as we’ve started with picking younger players with bright futures – and then we’ve got to look at the
development system underneath as well.

“It’s not a simple equation that the Wallabies are bad. Let’s be serious about this. The Wallabies are not where they need to be but underneath that we need to
fix the system to go forward. And if we can do that now we’ve got a rosy future.”

Sports opinion delivered daily
</item 10>

<item 11>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Argentina completely and utterly dominated the first 45 minutes, but have precious little to show for it.

The hosts pummeled Carlos Coronel's net, and while the Red Bulls goalkeeper has made a few saves, it's mostly been Argentina's wastefulness that has seen them
fail to follow up their early opener. Rodrigo de Paul hit the post, Alexis Mac Allister probably should have had a penalty, and Nico Gonzalez has missed a few
great chances from the left.

Overall, Lionel Scaloni will be pleased with his side's performance, but the match hangs in the balance on the scoresheet. In reality, thought, Paraguay need
vast improvement if they are to snatch a result after being meticulously carved to pieces.
</item 11>

<item 12>
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
</item 12>

<item 13>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“There were a few tears and I think that’s awesome to see, players showing vulnerability, just opening up about how much this tour and this sort of game means.



“There were some (people) saying this is probably one of the most important games of All Blacks history. (The) All Blacks went through a bit of a hard path to
get here, media, friends, family even criticising us and all that, and we stuck together. We weren’t even said to be in the final and now look at us, we’re
here.

Advertisement

“Different players take it differently and some of the older players probably were hit a little bit harder and it was good to see just a bit of emotion and the
want from them. Seeing that gives all the other players an extra two, three per cent to think ‘OK, we are going to the well and we’ve got to give it our all’.”

The All Blacks will experience a changing of the guard next season.
</item 13>

<item 14>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
“Morgan’s England team was a finely tuned machine with everyone knowing their role and most of them at the top of their form… all Morgan had to do was pull the
strings of a great unit of cricketers and he did it well, but my Mum could have captained that lot.

“On the other hand Jos has a dysfunctional squad of players. Poorly selected, poorly prepared, not a settled team, many not sure of their roles, patchy form,
confidence wobbly and to cap it all England’s best two and most influential white-ball players have not been available.”

Boycott’s voice was added to by former England Test captains Nasser Hussain and Michael Atherton, who have both decried the sudden decline of the team’s
previously dominant white-ball set-up, having entered the World Cup as reigning ODI and T20I champions.
</item 14>

<item 15>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Argentina completely dominated Paraguay from start to finish, both with and without Lionel Messi, and picked up three points in the process, although it was a
bit more nervy than they would have liked at the end.

With Argentina pummeling their CONMEBOL World Cup qualifying opponents in Buenos Aires from the opening whistle, Nicolas Otamendi's fourth-minute goal seemed to
foreshadow a coming onslaught. Yet that was all Argentina would get, and while it was enough for a 1-0 victory, they could have had many, many more.

Lionel Messi came off the bench shortly after halftime, but even he was powerless to change the narrative, hitting the post twice including once directly off a
corner.

Lautaro Martinez, Rodrigo de Paul, and Nico Gonzalez were all very active in the final third and had multiple chances, but none were able to find the
all-important second goal to kill off the match.
</item 15>

<item 16>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
“That judgment will be decided by the 2027 World Cup.”

He said the team – including coaches and players – have already completed their own review of the World Cup disaster. The next one would be “a formal review to
be done by independent consultants” and he was ready to participate.

“I didn’t come back to Australia to have a holiday – sit down at Coogee Beach, eat fish and chips, have a nice flat white. It was always going to be a battle.
When you’ve got a team that hasn’t done well for a long period of time, it’s always a battle.”

Advertisement

Jones said it was crucial that Rugby Australia got alignment in the code from grassroots through to the top team as it looks to rebuild it’s international
standing. Australia, a two-time world champion, is currently ranked 10th in the world.

“You just got to watch the quarterfinals at the weekend. We’re not at that level.
</item 16>

<item 17>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The Springboks have done it tough, too, having already faced two of the tournament favourites on their way to the semi-finals. Comfortable defeats of Scotland,
Romania and Tonga couldn't mask the disappointment of the 13-8 loss to Ireland, which consigned South Africa to second in Pool B. However, with Ireland sent
home by the All Blacks in the quarters, the Bokke showed their World Cup mettle to edge the hosts 29-28 at the Stade de France, sneaking past Les Bleus in one
of the best games of rugby you're ever likely to see.
</item 17>

<item 18>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
In the wake of the Wallabies’ World Cup flop, the Queensland Rugby Union have reaffirmed their position that they are committed to moving to a centralised
high-performance model but won’t follow the NSW Waratahs by handing over the keys to Rugby Australia.

They also expressed their strong support for Rugby Australia to conduct a “fully independent and transparent top-to-bottom review of the Wallabies program and
how it can be improved” in light of the historic first World Cup pool exit.

They added: “It should also provide a clear view of what is needed in a leadership and governance framework to ensure we put the right people, in the right
roles, in the right system.”

Facebook Twitter Whatsapp Reddit Email Share

The QRU has also expressed their desire for Rugby Australia to reinstate the annual $1.7 million in funding to each of the Super Rugby franchises that was
withdrawn following the Covid pandemic to help keep the game afloat.
</item 18>

<item 19>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
PARIS – The All Blacks have downplayed Rieko Ioane’s gestures towards Ireland fans and retiring great Johnny Sexton, as the three-time world champions said they
have learned to “respect Argentina” ahead of their semi-final on Saturday (6am AEDT).

It comes as Michael Cheika embraced the underdog status and said he was hoping Los Pumas “inspired” Argentina “not by our results but by our ambition”, as the
former Wallabies coach readies for another clash against the All Blacks and a second semi-final.

Facebook Twitter Whatsapp Reddit Email Share

Ioane copped a spray from Sexton following their 28-24 nail-biting win over Ireland.

It came after the lethal back cupped his ear at Irish supporters at the Stade de France after the final whistle, as well as putting his finger to his lips in a
‘shush gesture’.
</item 19>

<item 20>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“To inspire is not about results, it’s not just one player who played well; no, it’s about our behaviour in general, our ability to be ready to come back from
difficult times and seizing our opportunities.

“There are not a lot of opportunities in life – us playing in a semi-final is even rarer and we want to have no regrets.”
</item 20>

<item 21>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
The QRU also expressed its satisfaction that RA was willing to discuss the annual $1.7m being reinstated, which had been withheld because of the Covid pandemic
to keep the game afloat.

Meanwhile, The Roar understands the ACT Brumbies have sent RA confirmation of solvency for the next 12 months.

RA was forced to take out a $40m loan from Ares to keep the game afloat.

Advertisement

While RA has around another $15m left to spend from the loan, the governing body is looking to borrow up to another $60m so they’re in the strongest possible
position to capitalise on the two major money-making events – the 2025 British and Irish Lions series and home men’s World Cup in 2017 – and succeed.
</item 21>

<item 22>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
Oct 8th, 2023

Oct 8th, 2023

S﻿AINT-ETIENNE: The Wallabies will have to endure a more tortuous Sunday night watch than even the worst rom com as their Rugby World Cup fate is decided by
Fiji and Portugal in Toulouse.

Australia will gather together in their plush Saint-Galmier hotel at 9pm local time (6am Monday AEDT) to view what will almost certainly be their World Cup
execution.

The Wallabies are currently sitting second in Pool C but a single bonus point would be enough for Fiji to leapfrog them and advance to the quarter-finals in
Marseille.

Watch all the action from Rugby World Cup 2023 on the home of rugby, Stan Sport.
</item 22>

<item 23>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
Minutes later and Kolisi was off, as he cannoned into Savea. He would later return, much to the anger of the New Zealanders in the crowd who were calling for
justice.

Sensing an opportunity to strike against 14 men, the All Blacks kicked for the corner on a couple occasions out wide. It came back to haunt them, as their
usually silky and composed skill sets failed them.

That much was true when the TMO intervened after Smith thought he had a try in his final Test, but an error at the lineout cancelled out the five-pointer.

Advertisement

Eventually the All Blacks did have a try, as Mark Telea, the All Blacks’ best and biggest attacking threat all night, managed to get on the outside and draw in
two defenders before offloading to Beauden Barrett. But Mo’unga missed the conversion to leave them trailing by a point.
</item 23>

<item 24>
Damien Penaud leads historic rout of Italy as France confirm progress in style (The Guardian, 2023-10-06T21:06:37+00:00)
But that wasn’t my decision.

“What needs to happen is that these players need to be treated with a bit of respect off the field … They [the Italian Rugby Federation] need to get some of
that stuff sorted. Quite honestly, I don’t know if I’d want to be involved unless that gets sorted. I respect these guys, these players and staff. They’ve gone
way beyond … and yet people will judge us on these last two games. It’s disappointing.”

Damaging setbacks have arrived with regularity for France, most notably Antoine Dupont’s fractured cheekbone. But the prop Cyril Baille had returned from injury
in the thrashing of Namibia, along with the estimable Jonathan Danty in midfield, and both started against Italy.

Any hint of home nerves was calmed by the try for Penaud that stemmed from the first attack. Uini Atonio and Thibaud Flament were both involved, busting through
tackles and creating a huge overlap for Penaud.
</item 24>

<item 25>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
The Melbourne Rebels are expected to follow.

RA chief executive Phil Waugh says building trust back between the governing body and states is essential to their plans.

Under fire Wallabies coach Eddie Jones also advocated for Australian rugby to move to a centralised model, believing the nation had fallen behind the rest of
the world.

“We’re a state-based union that’s run by the states, the national union wants to assume more responsibility and when things were going well, no one questioned
that,” Jones said.

Advertisement

“But over the last period of time, they haven’t been going so well and this tournament’s probably been the icing on the cake where it’s got to a stage where I
think people, most people probably think there’s time for change. But it’s not going to go easy.

“[It’s] one of the key issues. We need to maximize the development of the players.
</item 25>

<item 26>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
In the open letter to their members on Wednesday morning, QRU chief executive David Hanham and chairman Brett Clark moved to update constituents on the reform
measures that RA is attempting to usher through the game.

Last week, The Roar revealed that the New South Wales Rugby Union was in the latter stages of handing over their license to RA.

By doing so, the NSW Waratahs, who were on the brink of insolvency, handed over their commercial and high-performance property.

Advertisement

The decision will see all male and female Waratahs players, including coaches and high-performance staff, as well as all commercial and operations staff, run
out of RA’s office.

NSW Waratahs chief executive Paul Doorn informed stakeholders of the decision last week.

RA is cognisant that not every Super Rugby franchise will follow suit immediately, but the governing body is hopeful that now one of the two biggest states has
agreed to hand over the keys, others will follow suit swiftly.
</item 26>

<item 27>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
None of those targets were achieved and Jones changed his tune about building for the future when he named a young team for the doomed Cup campaign – where his
team became the first Australians to bow out of a RWC in the pool stages.

He finished the year with a 2-7 win loss record – the only two victories coming against lowly Georgia and Portugal.

Advertisement

On Tuesday he said he no regrets with leaving experienced players such as Quade Cooper and Michael Hooper out of the side for the tournament.

“Everyone’s got their own judgment and they’re allowed to have that judgment,” he said. “My choice was that I decided to go with youth. And I stand by that
because I think we’ve got a a lot of good young players who are really going to benefit from that experience at the World Cup, and going forward to the 2027
World Cup.”
</item 27>

<item 28>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
The QRU confirmed that on Wednesday morning.

Advertisement

“The QRU is committed to working with Rugby Australia to lead reform of our game at a high-performance level, and help Rugby Australia develop an improved game
through high-performance centralisation,” the statement read.

“Centralisation” would include a national player-contracting model, and alignment of key high-performance staff, such as coaching, strength and conditioning,
and medical to operate as one national system.”

The letter pointed to RA’s involvement in Les Kiss’ appointment as Reds coach, with the former State of Origin winger turned coach, who has spent the best part
of two decades coaching in Ireland and England, replacing Brad Thorn.

As The Roar has previously reported, the QRU stopped short of wanting to hand over its commercial property.

“While the QRU is very supportive of high-performance alignment, it will not agree to any proposal on centralising commercial or corporate functions,” the
letter read.
</item 28>

<item 29>
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
</item 29>

<item 30>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
While Ireland will rue the 37 phases that ended in heartbreak, they may just go down in New Zealand rugby history. Tired bodies made tackle after tackle and not
a single penalty was given away.

The delight at the final whistle showed this wasn’t just any quarter-final victory and while Ireland ponder just what they have to do to finally win a World Cup
knockout match, the All Blacks can dream of lifting the Webb Ellis Cup for a fourth time. The pain and ecstasy of sport writ large.
</item 30>

<item 31>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
He and Franco Mostert pack down together in the second row, while hooker Bongi Mbonambi, utility Deon Fourie, veteran Duane Vermeulen and the devilish Kwagga
Smith will all be looking to back up superb quarter-final performances to take the Springboks to a second consecutive World Cup final.

South Africa starting XV: Damian Willemse, Kurt-Lee Arendse, Jesse Kriel, Damian de Allende, Cheslin Kolbe, Manie Libbok, Cobus Reinach; Steven Kitshoff, Bongi
Mbonambi, Frans Malherbe, Eben Etzebeth, Franco Mostert, Siya Kolisi, Pieter-Steph du Toit, Duane Vermeulen.

South Africa replacements: Deon Fourie, Ox Nche, Vincent Koch, RG Snyman, Kwagga Smith, Faf de Klerk, Handre Pollard, Willie le Roux.
</item 31>

<item 32>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
In short, this semi-final requires a huge step up in quality for Borthwick's side, and that's why we think that the Springboks will be victorious at the Stade
de France. South Africa have already shown they can match, and beat, the world's best, and we think that they'll do the latter in Saint-Denis.
</item 32>

<item 33>
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
</item 33>

<item 34>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
Advertisement

Robertson will have an almighty task on his hand, with multiple generational players including Dane Coles, Sam Whitelock, Brodie Retallick and Aaron Smith to
call time on their international careers. Others like Ardie Savea, Mo’unga, Beauden Barrett and Shannon Frizell will also head to Japan.

While Foster didn’t want to take anything away from the Springboks’ win, he said the game had some issues to sort out regarding the officiating.

“I’m proud of our guys, to go down to that red card so early and fight our way back and give ourselves a chance is pretty special,” the departing All Blacks
coach said.

“I don’t think it went wrong in any clear [place on the pitch], it was a real arm wrestle, both teams have their moment, incredibly proud of the way we fought
and to get within a whisker of pulling it off is heart-breaking.
</item 34>

<item 35>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
Seriously, the amount of time he puts into effort, into his staff, into the players, always has time for us ... and I think fans don't see that," he said.

"At the end of the day, we didn't get the results, you know, we didn't perform the way we wanted to as a team and it's not on Eddie - it's on all of us.

"So I understand what the fans feel and they're just passionate and it's awesome to have that. But no-one realises what goes on inside. No-one realises the
things that we're working on tirelessly, to work hard, but especially him and the staff, f--ing work so hard, bro.

Samu Kerevi during a Wallabies training session at Stade Roger Baudras. (Getty)

"They're away from families, they're sacrificing as well and … I don't read into the comment stuff, but the young boys, I hear a lot that's going on, social
media and stuff.
</item 35>

<item 36>
Another cruel comeback sees Newcastle perish in group of death (The Independent - Sports, 2023-12-13T22:16:07+00:00)
But it is over, and they are left to wonder about what might have been.
</item 36>

<item 37>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Argentina are playing champagne football. 🇦🇷 pic.twitter.com/y2u5IqqUPy — Roy Nemer (@RoyNemer) October 12, 2023

25th min: Julian Alvarez has a shot saved from a very tight angle by Carlos Coronel, conceding a corner. The chance was threaded in beautifully by his strike
partner Lautaro Martinez. The set-piece is looped well off target by the head of Cristian Romero.

21st min: The first yellow card of the game is shown to Junior Alonso who comes flying into a challenge with Alexis Mac Allister and catches him high on the
ankle. A very bad tackle — not quite worthy of a red, but it's not far off either.

16th min: Ouch, Lautaro Martinez tries to latch onto a wonderful vertical flick from Enzo Fernandez up the gut, but he's just a split-second too late, and he
goes sliding into the ribs of Paraguay goalkeeper Carlos Coronel.
</item 37>

<item 38>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England vs. South Africa betting odds, prediction

The last time these two sides met was a tumultuous period to say the least, particularly for English rugby. South Africa's dominant 27-13 victory at Twickenham
proved to be the last game in charge for Eddie Jones, the coach who had taken England to the 2019 World Cup final. In his place, Steve Borthwick was hired, and
he's just one game away from repeating that feat.

England's form at this tournament has been perfect — at face value. Argentina, Japan, Chile, Samoa and Fiji have all provided highlights — Los Pumas are also in
the semi-finals — and England have beaten all of them. However, only Japan and Chile were defeated with bonus points, and the narrow wins over both Samoa and
Fiji in their last two games showed that England are certainly gettable.
</item 38>

<item 39>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
But I don't think people really understand demons, everyone's fights in between.

"Eddie's a human as well and at the end of the day it's a results-driven sport, you know, we need results to get that. But he's committed to us and we love that
about him.

"He hasn't thought about going home, he's thought about how we can contribute best and that's what we've done these last three days. I feel for him, you know,
like as a head coach, you maybe can't show that side of himself.

Eddie Jones and Will Skelton talk at Stade Roger Baudras. (Getty)

"But he's been so strong for us and gives us strength to just be positive. And again, we put ourselves in this situation."

Wide World of Sports also asked Kerevi to deliver a message to disillusioned Wallabies fans.﻿

"I think at home there's so many sports going on and some saying rugby's dying, but rugby's massive here overseas," he said.
</item 39>

<item 40>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
So the team understands that and we have our belief in anything's possible at the World Cup.

READ MORE: ﻿'Blame myself': Eddie reveals great World Cup regret

READ MORE: Referee ripped after Junior Paulo's dominant KO

LIVE COVERAGE: All the latest updates from Bathurst 1000

Mark Nawaqanitawase runs the ball during a Wallabies training session at Stade Roger Baudras. (Getty)

"But also, we understand how strong Fiji is. And we're proud for them as a proud nation. For myself, I'll be proud of it either way it goes. We'll be ready if
it comes our way."

Kerevi was photographed holding his face in disbelief in one of the defining images of the World Cup ﻿after the Wallabies were pumped 40-6 by Wales in Lyon.
</item 40>

<item 41>
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
</item 41>

<item 42>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
“Some of those cricketers on that park today would go in your best ever England white ball XI. It’s just been a bridge too far for most of them.

“What I don’t like is giving players a cop out… when we win the 50-over World Cup and the 20-over World Cup, aren’t they great? We’re brilliant. And when the
wheels come of it’s the structure. It’s the structure of English cricket, we’re a disgrace.

“We play 20-over cricket, we play 100-ball cricket, we don’t play enough 50-over cricket – how much 50-over cricket domestically has Virat Kohli played? Or
Heinrich Klaasen or anyone out here?

“That’s what’s made this great side over the last six years, travelling round the world playing T20 franchises. It’s such a lame excuse.”
</item 42>

<item 43>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
“Firstly, you’ve never heard us say we’re favourites,” he said.

‘We know these games are do or die. We’re going to have to improve our performance. The stage gets bigger at this time in the tournament and you’ve got to grow
your game. That’s our strong focus. The best team on the night wins – Argentina has done that to us.”

Cheika however was happy to remind the All Blacks who were favourites.

“”New Zealand has always been an example in rugby, a benchmark. They make you think of high-level skills in a really open game but there are always threats in
the lineout, mauls and rucks,” said Cheika, whose Wallabies lost in the 2015 final.

“You have to take the moments you are in the game and see them coming at you at different times.

Advertisement

“In the game in 2020 there were things you could pinpoint.
</item 43>

<item 44>
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
</item 44>

<item 45>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
The victory was the perfect sign-off for the Springboks’ management, with director of rugby Rassie Erasmus set to follow head coach Jacques Nienaber out.

The duo will go down as the greatest coaching duo in rugby history, having led the Springboks to back-to-back World Cup triumphs.

“We came a long way with the players. We always planned for this World Cup, since 2018,” Nienaber said.

“Massive credit to our fans. They were special from South Africa, 62 million people, the videos we had to play from them. It was unbelievable.”

Veteran back-rower Duane Vermeulen will also call time on his illustrious career.

All Blacks coach Ian Foster’s tenure in charge has also come to an end, with Scott Robertson to take over following a bumpy four years in charge.
</item 45>

<item 46>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Gomez does well to dispossess Messi, but the Argentine is first back on the loose ball, expertly shielding the ball with his body, and Gomez obliges by barging
in from behind.

With the set-piece, Messi hits the woodwork! Carlos Coronel is helpless to do anything but watch, rooted to the spot, and he sees it sail by him and strike the
outside of the left post!

Gio Lo Celso comes on for Argentina to replace Rodrigo de Paul, who had a very good match.

89th min: As this game sputters towards a conclusion, Emiliano Martinez has set a new national team record for a shutout streak, crossing 609 minutes of play
without conceding a goal. It's an impressive mark nonetheless, but in fairness, he's had almost nothing to do tonight, making just one (admittedly impressive)
save.

Argentina have not conceded a single goal since the 2022 World Cup final. Four minutes of stoppage time are shown.
</item 46>

<item 47>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
Sam Cane – the All Blacks captain – was forced to watch the match from the sidelines for 53 minutes after having his yellow card for a high shot on Jesse Kriel
upgraded to red. His opposite in the number seven jersey, Siya Kolisi, was later shown a yellow card for a high shot on Ardie Savea, but returned.

Cheslin Kolbe – the magnificent winger who sealed their World Cup triumph four years earlier against England – meanwhile couldn’t bare to watch from the
sidelines after being shown a yellow card for a deliberate knockdown in the 73rd minute.

Advertisement

It was that kind of evening, where the margins were tiny and the outcomes almighty.

“There are no ways I can explain it,” Kolisi said.

“I want to give credit to the All Blacks. They took us to the end, they took us to a dark place. It shows what kind of team they are, to fight with a man down
from early in the game.
</item 47>

<item 48>
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
</item 48>

<item 49>
Damien Penaud leads historic rout of Italy as France confirm progress in style (The Guardian, 2023-10-06T21:06:37+00:00)
“It’s easy to congratulate Damian for his attacking performance,” Galthié said. “But it’s also easy when you have a team like that working for you. Players that
become great are served by their teams; there’s always a balance.” The former France scrum-half added: “We’re happy to qualify … We’ll recuperate together,
celebrate qualification, then turn to the quarter-final. For us, after a World Cup final against New Zealand [on opening night], our game next Sunday will be a
second final.”

For Kieran Crowley, Galthié’s opposite number, his Italy tenure ends in profound disappointment, although crushing defeats by the All Blacks and now the hosts
are offset by victories against Namibia and Uruguay.

“I’ve said before, I’d like to take this team forward,” he said. “I think the next World Cup is our cycle … we’ll get to the next tournament, and a lot of these
players will have 60, 70 caps. But that wasn’t my decision.
</item 49>

<item 50>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
But after initially moving back into the coaching fray with the Blues in Super Rugby, he joined the All Blacks ahead of last year’s Rugby Championship as New
Zealand Rugby shook the All Blacks coaching team up following their historic series loss to Andy Farrell’s Irish.

Schmidt’s expertise proved telling, with he and Jason Ryan transforming the All Blacks and helping Ian Foster’s side reach the final of this year’s World Cup
after stunning wins over Ireland and Argentina in the knockout stages.

Importantly, the 58-year-old, The Roar understands, is interested in the Wallabies coaching job.

Dan McKellar, Stephen Larkham, Andy Friend and Ian Foster remain interested in the role.

Should RA move on him, Schmidt’s backing would be largely supported by the Super Rugby franchises, with several sources telling The Roar that the experienced
coach is exactly the type of person the Wallabies need.
</item 50>

<item 51>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
A missed Richie Mo’unga conversion following Beauden Barrett’s 58th minute try – the only five-pointer of the match – ultimately proved the difference, as
Handre Pollard’s four three-pointers led the Springboks to World Cup glory.

The All Blacks had their chances, but their usual safe handling, a feature throughout the World Cup after their first-up loss to France, evaded them against the
Springboks’ blitz defence in the wet conditions.

All night the All Blacks’ timing was off and the forced ball from Savea out wide just as the men in black saw space and looked to open up the Springboks in the
80th minute proved to be killer blow.

Wayne Barnes, who was booed relentlessly, called a scrum reset as the seconds ticked by.

Advertisement

Moments later, their hearts sunk as the Springboks managed to win the scrum and then hang onto the ball.
</item 51>

<item 52>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
The biggest error was around the corner, as Kriel stepped back into the defence and Cane failed to drop his body height and clocked the centre in the jaw. He
didn’t return.

Advertisement

The Springboks took a 12-3 lead after 34 minutes when the All Blacks once again failed to release on their goal line. Pollard licked his lips and said ‘thanks’.

The All Blacks managed to cut the deficit to 12-6 after 37 minutes when Eben Etzebeth was caught offside and denied his opponents some quick ball.

Foster’s men breathed a sigh of relief when Kolisi blew a chance out wide when he tucked the ball under his arms and went himself after Beauden Barrett failed
to collect a high ball in the 42nd minute.

Minutes later and Kolisi was off, as he cannoned into Savea.
</item 52>

<item 53>
Watch Argentina vs Paraguay live stream, TV channel, lineups, betting odds for World Cup qualifying match (Sporting News, 2023-10-12T21:11:00+00:00)
SGT New Zealand Fri, Oct. 13 11 a.m. NZST

Argentina vs Paraguay starting lineups & team news

Argentina head coach Lionel Scaloni named a bloated 36-man squad for the matches against Paraguay and Peru this month, but Angel Di Maria, Paulo Dybala, and
Juan Foyth were forced to withdraw due to injury. Lisandro Martinez is also unavailable after having to undergo another foot operation.

Now that Inter Miami's playoff run is officially over, Argentina can feel a little better about deploying Lionel Messi at full 90-minute use despite his recent
fitness issues, knowing his club season is nearly over and he'll have months of rest upcoming. Messi missed the last match against Bolivia, and missed four
Inter Miami games after, but he returned for a half-hour of action in their final fixture before the international break. However, he doesn't start, but is
reportedly fit enough to come off the bench for most or all of the second half.
</item 53>

<item 54>
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
</item 54>

<item 55>
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
</item 55>

<item 56>
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
</item 56>

<item 57>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Cover 7 | Monday A daily NFL destination that provides in-depth analysis of football’s biggest stories. Each Monday, Mike Sando breaks down the six most
impactful takeaways from the week.

The San Francisco 49ers entered Week 6 riding a 15-game winning streak and looking unstoppable with a young quarterback, Brock Purdy, who was making the game
look easy. They exited with a 19-17 defeat at Cleveland, injuries to key players and, for the first time this season, a question mark or two.

Advertisement

With the previously unbeaten Philadelphia Eagles also losing ugly Sunday, this edition of the Pick Six column addresses key questions facing Super Bowl
favorites with one-third of the 18-week regular season nearly complete.

So far this season, we’ve seen the Miami Dolphins set NFL records for offense, only to lose by four touchdowns in their lone game against a true contender.
We’ve seen the Kansas City Chiefs win with defense, sometimes in spite of their offense. We’ve seen the Buffalo Bills incur devastating injuries.
</item 57>

<item 58>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
"Last week 45,000 fans (versus Portugal), last night (60,000 in Lyon). Rugby's such a small aspect in terms of Australians but our fans have turned up here,
we've seen them and spoken to them and their unconditional love for us, the fact that they came 20,000km to get here and support us, has been awesome.

"My message is just to understand that we're working extremely hard, there's no stone unturned, especially the youth in this team, to leave the jersey in a
better place and just to stick with us.

"I know the results haven't been coming our way the last couple of years but when it does turn, that's what we're building towards.﻿"
</item 58>

<item 59>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The last time Cheslin Kolbe played England at a Rugby World Cup... 🇿🇦 🏆#RWC2023 | #ENGvRSA pic.twitter.com/Riw2uaSaKa — Rugby World Cup (@rugbyworldcup) October
19, 2023

Both sides' journeys have seen twists, turns, and bumps in the road, but it's led them both to within 80 minutes of a Rugby World Cup final. England will no
doubt be looking to echo the spirit of 2019, when they beat the All Blacks 19-7 to reach the final, but in South Africa, they face a side who have taken their
game to new heights just to reach the semis.

MORE: 2023 Rugby World Cup final: Details, kick-off time and every team's route

What time does England vs South Africa kick off?
</item 59>

<item 60>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
RA is also in the early stages of looking to renew its next broadcast deal, with the governing body hoping to at least double its current deal with Nine
Entertainment/Stan Sport, which is seeing them paid $29m per year until the end of 2025.
</item 60>

<item 61>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“We don’t want to talk too much about post-game stuff, but that was a tough one, probably the toughest I have had as a coach. He took it the way I expected him
to. He’s a champion.”

Papali’i said the All Blacks were intent on sending their leaders out on a high.

“There’s a few leaders in the team, it’s going to be their last game in the jersey, so it’s a big game for us and even the other players that are staying,
knowing that we want to send those boys out on a high,” he said.

“How much they put into the jersey. We are standing on their…what’s that quote? It’s standing on the shoulders of giants.”

The generational change isn’t just occurring on the field but off it, too.

Advertisement

Foster will depart the All Blacks set up after 12 years, the last four as head coach.
</item 61>

<item 62>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
TV channel Streaming USA — Peacock Canada TSN TSN+ UK ITV ITV X Australia Channel 9 Stan Sport, 9Now New Zealand Sky GO Sky GO India — Fan Code Hong Kong BeIN
Sports BeIN Sports Malaysia BeIN Sports BeIN Sports Singapore BeIN Sports BeIN Sports

MORE: Every single team to have lifted the Rugby World Cup trophy

England vs South Africa lineups, team news

"The start is crucial, the end is vital..."



- Steve Borthwick explains changing 3️⃣ of his England starters for Saturday's Rugby World Cup semi-final, from Liam Heagney 👨‍💻 in Paris. #RWC2023 #ENGvRSA
#EnglandRugby
</item 62>

<item 63>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
Foster will leave his role after the World Cup, to be replaced by all-conquering Crusaders boss Scott Robertson, but he can still exit with the greatest prize
of all. And frankly, who would bet against the All Blacks now?

The opening-night nerves against France seem a long time ago and they ultimately had too much for Ireland in this simply incredible, back-and-forth Test match
that may genuinely go down as one of the best of all time.

After early penalties from Richie Mo’unga and Jordie Barrett, they scored a scintillating try through Leicester Fainga’anuku, who was only in the starting XV
due to Mark Telea’s breach of team protocol earlier in the week. Beauden Barrett expertly dinked over the top, collected his own chip and when the ball was
spread wide, Fainga’anuku and Rieko Ioane neatly combined to send the former over in the corner.
</item 63>

<item 64>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Advertisement

After a year in where the Wallabies’ detail and nous was clearly absent on the field, the highly detailed Schmidt would help bring rugby smarts to the
Australian program.

Indeed, the manipulation and ability to play both sides of the ruck was clearly visible by Ireland during his Irish tenure.

What’s more, should RA go down a path of bringing Schmidt into the fold, his own relationship with Ireland’s long-time director of high-performance David
Nucifora could see the Australian reconsider his position with the governing body.

Last year Nucifora told The Roar he would unlikely consider returning to RA when he leaves his post with Ireland following next year’s Olympics.

“I don’t think it’d be wise to revisit,” Nucifora said.

“I think that you have a go at something once, you do your best and then if it’s not good enough or it didn’t work you move on, do something else.
</item 64>

<item 65>
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
</item 65>

<item 66>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Yet in the end, Emiliano Martinez had himself another clean sheet as Paraguay failed to produce more than a single quality attacking chance.

The shutout gave Dibu an Argentine national team record, passing 609 minutes without conceding a goal, as Argentina have not been breached since Kylian Mbappe's
dramatic 118th minute equalizer in the 2022 World Cup final. Additionally, Lionel Scaloni has still never been beaten in World Cup qualifying as Argentina head
coach.

The Sporting News followed the Argentina vs. Paraguay match live, providing score updates, commentary and highlights as they happened.

MORE: Where to watch Argentina vs. Paraguay in World Cup qualifying

Argentina vs Paraguay final score

Score Goal scorers Argentina 1 Otamendi (4') Paraguay 0 —

Kickoff: 8 p.m. local (7 p.m. ET / 4 p.m.
</item 66>

<item 67>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Before he can get a shot off, he's bumped to the ground, and replay shows he has a clear grievance with the no-call! VAR does not intervene.

ALEXIS MAC ALLISTER, STOP THAT.pic.twitter.com/wlShqa6mqR — Roy Nemer (@RoyNemer) October 12, 2023

34th min: Alvaro Campuzano is whistled for a foul after hip checking Nicolas Tagliafico to the ground, handing Argentina a dangerous free-kick just inside the
attacking third. The call by the official sends both Campuzano and Paraguay manager Daniel Garnero into a frenzy, as neither agree with the decision, but it's
the right call for overly aggressive body positioning.

The chance is wasted, however, as it is easily cleared and Argentina must reset.

30th min: Chance, Argentina! Lautaro Martinez has a vicious shot agonizingly deflected wide!
</item 67>

<item 68>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
We need to maximize the development of the players. At the end of the day, that’s the key to success and alignment of your resources, for a small country.

“Because Australia, if you look at the history of rugby, when we were amateur, we were probably ahead of the game because we had the influence of rugby league.
We had the influence of AFL and our players were more professional than the rest of the world.”

He added: “I know what’s going on and we need to get a better system of developing players. We need to develop better players and we need to develop a larger
batch of better players and that only comes through changing the whole system.”

While all five Australian Super Rugby sides fundamentally are on board with regards to moving to a centralised high-performance program, The Roar added that
some states, including the ACT Brumbies as well as the QRU, don’t want to give up their commercial assets.
</item 68>

<item 69>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England replacements: Theo Dan, Ellis Genge, Kyle Sinckler, Ollie Chessum, Billy Vunipola, Danny Care, George Ford, Ollie Lawrence.

The Springboks are unchanged from that epic quarter-final with France, having all played their part as Jacques Nienaber's side hung on in the second half. The
contribution of backs Kurt-Lee Arendse, Damian de Allende and Cheslin Kolbe saw all three cross for first-half tries, with Kolbe's coming off the back of a
charged-down conversion from France's Thomas Ramos. Preventing those two points proved crucial in the end.

The key contribution, though, came from Eben Etzebeth, who bulldozed his way over to score the only five-pointer of the second half and wrestle the game back
into South African hands.
</item 69>

<item 70>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
41st min: Paraguay are getting extremely frustrated with their lack of attacking possession. They get into the final third for the first time in ages, but with
no support, Antonio Bareiro blasts a hopeless effort from a long way out, the shot blocked by his defender who's all over the lone Paraguay striker like white
on rice.

38th min: Chance, Argentina! A deep delivery from way out on the right touchline meets the head of a leaping Nico Gonzalez at the far post, but it's saved well
by Carlos Coronel! The header was from a very tight angle, but the Red Bulls goalkeeper still had work to do as he reacted quickly to stick out his left foot
and make the reflex stop.

36th min: Chance, Argentina! The referee very dramatically waves away a penalty shout from Alexis Mac Allister! A corner is cleared to the Liverpool midfielder
at the top of the penalty area, and he does brilliantly to dribble through multiple defenders.
</item 70>

<item 71>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
In March last year Graham Arnold seemed a dead man walking – the Socceroos World Cup qualification campaign was lurching towards failure and the manager was
being battered from all sides.

Arnold’s team had finished seven points off automatic qualification for Qatar after a shocking run of one win in seven games left them facing a Asian playoff
against United Arab Emirates and then another against Peru to make the finals.

While rumours swirled and ex-players and fans bayed for his blood, Football Australia took the pragmatic course and stuck by Arnold. It likely had more to do
with the lateness of the hour and financial considerations than any overwhelming faith in Arnie to run around the fortunes.

Facebook Twitter Whatsapp Reddit Email Share

But he did just that with qualification – and an excellent run at the finals – bowing out to the eventual champions Argentina and after nearly taking them to
extra time.
</item 71>

<item 72>
England thrash Scotland but cannot avoid Team GB Olympic heartbreak (The Guardian, 2023-12-05T21:56:19+00:00)
If there was any doubt about Scotland’s intentions against England, with Bronze calling accusations that Scotland players might feel conflicted with the
possibility of Olympic qualification on the line for them as much as for England “the rudest thing I’ve ever heard” and Scotland’s captain, Rachel Corsie,
calling it “disrespectful” and “outrageous” those were put to bed, temporarily at least, in the opening three minutes, with Bronze clattered into late twice as
Scotland looked to make an early impression.

If England had played well for the opening 10 minutes against the Dutch only to be sucker-punched on the counter in the 12th minute, against Scotland there was
no risk of that. Instead, the visiting team took the lead to swing the pressure on to the Dutch, Mead’s corner sent in by an unmarked Greenwood.

Moments later Lisa Evans was down in the box under pressure from Bronze but appeals for a penalty were waved away with the right-back having not caught the
forwards legs.
</item 72>

<item 73>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
Sign up to our free sport newsletter for all the latest news on everything from cycling to boxing Sign up to our free sport email for all the latest news Please
enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read
our privacy notice Thanks for signing up to the

Sport email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

This time was supposed to be different. This Ireland team were different… are different. But, come the end, the result was the same. On quarter-final weekend,
for the eighth time in 10 Rugby World Cups, Ireland head home defeated and the wait for a first-ever knockout win goes on.
</item 73>

<item 74>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Bareiro fouls Romero and gets away with it, and Romero reacts later off the ball by bumping the Paraguay forward, who throws himself dramatically to the turf.
It's the reaction which earns the booking, while Romero is lucky to get away with his brainless moment.

Kickoff: They've begun the second half, and while Lionel Messi has not entered the pitch yet, he's warming up on the touchline and likely to come on soon.
Alexis Mac Allister is fouled from behind just seconds after the restart.

Robert Rojas has come on for Paraguay in the first substitution of the match, entering at right-back for Ivan Ramirez.

HALFTIME: Argentina 1-0 Paraguay

Somehow, some way, Nicolas Otamendi's fourth-minute goal remains the only successful strike of this game so far. Argentina completely and utterly dominated the
first 45 minutes, but have precious little to show for it.
</item 74>

<item 75>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
Every match ad free, live and on demand in 4K UHD

The most likely outcome is that Fiji wins easily and the Wallabies will be pushing confirm on their off-season holiday plans.﻿

"We've only got ourselves to blame, we put ourselves in this position," Wallabies centre Samu Kerevi, who was born in Fiji, said after an open training session
in front of 1000 keen locals at the Andrezieux-Boutheon rugby club on Saturday.

"We'll sit down as a team and watch together on Sunday night. I think we understand the situation ... our results and the way we played, we put ourselves in
this position. So the team understands that and we have our belief in anything's possible at the World Cup.
</item 75>

<item 76>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
They put us under so much pressure.

“Credit to my boys too for the fight. I am just grateful we could pull it off.”

Speaking of mighty, Pieter-Steph du Toit – the 2019 World Rugby player of the year – played his best match in four years as he spearheaded the Springboks’
physical onslaught. The back-rower made 28 tackles, each as bone-crunching as the next.

Advertisement

By claiming their third one-point knockout victory in a row, the Springboks moved into uncharted territory by leapfrogging the All Blacks by sealing their
fourth World Cup. It was the second straight tournament the Springboks won in France, having beaten England at the same venue in 2007.
</item 76>

<item 77>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
NEW PODCAST! Michael Atkinson and Sam Worthington preview a thrilling final round of pool games to decide the quarter-finalists at the Rugby World Cup in
France﻿

He wanted to regain form at club level ﻿with Urayasu D-Rocks but his hand remained up for Test selection.

"I chose to leave Australian shores and understand the ruling around that and what it holds," he said.

"I've got to perform for my club first and if you get the call up, you get the call up. If you don't, I'll take my three months break and enjoy that."

Kerevi is an unabashed Jones fan.

Eddie Jones with fans during a Wallabies training session at Stade Roger Baudras. (Getty)

"He's up at 4am working hard. We catch him in the gym, he's working hard and I swear he doesn't even sleep.
</item 77>

<item 78>
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
</item 78>

<item 79>
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
</item 79>

<item 80>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
It’s been far from smooth, having overseen their historic defeats to Argentina as well as the 2-1 series loss to Ireland on home soil.

A famous 35-23 win over the Springboks at Ellis Park saw him save his career, with Scott Robertson being lined up on the side.

But it wasn’t enough for New Zealand Rugby to back him before this year’s Test season got underway, with Robertson announced as Foster’s successor.

“Am I at peace?” Foster repeated a question during his World Cup team announcement when asked about where his mental state was ahead of his last Test in charge
of the All Blacks.

“Yeah, I’m reasonably peaceful. I love the word but I’m not sure I’m peaceful right now.”

Advertisement

He added: “I’ve just started to get my back nice and straight from last year and now you’re trying to make me hunched over again!

“It’s the life of this business, it’s a tough game.
</item 80>

<item 81>
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
</item 81>

<item 82>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
The 30-year-old is undoubtedly one of the world's finest centres in peak form but injuries conspired to limit his potential in France.﻿

Samu Kerevi of Australia reacts after the record loss at the Rugby World Cup. (Getty)

"I don't think I performed where I wanted to perform at a World Cup," Kerevi said.

"I hold myself to high standards and I just felt like I could have helped the team more in certain aspects ... I've been out of the game for nine months but I
felt good coming into it ... I just felt like I was blowing a bit, but felt good towards the back end of a couple of games. But again, it's not where I want my
performance to be.

Despite being dropped for Lalakai Foketi for the pool win against Portugal, Kerevi said he had no thoughts of following fellow Fijian-Wallaby ﻿Marika Koroibete
into international retirement.

NEW PODCAST!
</item 82>

<item 83>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
That looked painful for the Red Bulls goalie, but he's ok to continue.

It's a bit surprising that Lautaro got away with no booking from that really dangerous decision.

13th min: Chance, Argentina! Oh no, it's a terribly bad miss from Nico Gonzalez. The Argentine winger breaks down the left, and with the Paraguay defense
lagging behind, he's got plenty of room to shoot across the face of goal from a somewhat tight angle.

The "shot" is so bad it's hard to tell if he was trying to put it on frame or square it for a teammate at the back post, but it's a tragic attempt either way,
as there was nobody anywhere close at the far stick to meet it.

9th min: Paraguay get their first attacking move playing long to the striker, but after a few nervy moments the Argentina defense recovers and gets back on the
ball.

4th min: GOAL! ARGENTINA!
</item 83>

<item 84>
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
</item 84>

<item 85>
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
</item 85>

<item 86>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
The penalty allowed Pollard to step up and bang over three points.

The All Blacks were on the backfoot much of the opening quarter and, after Ethan de Groot was pinged for not rolling away, the Springboks moved further ahead
after 12 minutes.

A wicked bounce following a lovely Jordie Barrett chip kick denied Savea a try soon after. But the All Blacks were on the scoreboard via Mo’unga moments later
as Barnes was playing advantage.

But the Springboks restored their six-point lead soon after as Savea was pinged for not releasing. Barnes later apologised, with the No.8 having a split-second
separation.

The Springboks’ rush defence caused the All Blacks problems all evening, as Damian de Allende and Kriel forced their opponents into errors. Will Jordan, the
impressive young outside back who scored eight tries, had a difficult night and was replaced midway through the second half.
</item 86>

<item 87>
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
</item 87>

<item 88>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Advertisement

“What’s changed back there that I could think that, what I can offer, can change what they’re doing still?

“I’ll always love rugby. I’ll always love Australian rugby. Could I see myself working in Australia again? I doubt it.”

But RA have been making moves to try and twist Nucifora’s arm.

While Nucifora was never in the running for the director of high-performance, it’s understood they are trying to get him on board as a consultant when he wraps
up with Ireland.
</item 88>

<item 89>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
This is the greatest Ireland team of all time, they may still even be the best team in the world but on this day, the quarter-final curse, not to mention an
inspired All Blacks side, ruled once more.

This defeat will hurt more than any other. Truthfully, since 1991 when Michael Lynagh’s late try handed Australia victory, Ireland hadn’t come close to ending
their last-eight hoodoo. Never mind winning one, they haven’t led a World Cup quarter-final since the first half of the 1995 defeat to France, a mammoth 10,353
days ago.

But here they had a legitimate chance to reverse the curse. Multiple times they got within a point of the All Blacks, only for their opponents to agonisingly
pull away once more. And then there were those 37 phases… the most painful finish in Irish rugby history.
</item 89>

<item 90>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
But Mo’unga missed the conversion to leave them trailing by a point.

That lone point would prove the difference, as the Springboks celebrated a remarkable victory by a remarkable side.
</item 90>

<item 91>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
“It’s the life of this business, it’s a tough game. It’s tough when you’re trying to get your performance right. It’s a tough game when people around you see
things differently, but we’ve learned a lot about ourselves.

“We take massive pride in the jersey and making sure we reach the levels we want to. The rest of that is probably a conversation for another day.

“It hasn’t altered our vision of where we want this team to be. I know we’ve surprised a few people but I don’t think we’ve surprised ourselves.”
</item 91>

<item 92>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The #Springboks are gearing up for another tough challenge this week with a semi-final clash against England 🇿🇦🏴󠁧󠁢󠁥󠁮󠁧󠁿#StrongerTogether #RWC2023
pic.twitter.com/HLSnnjWggB — Springboks (@Springboks) October 18, 2023

England are the only side still competing who are yet to lose a game, and that will stand them in good stead coming into this semi-final. Dominant wins over
Argentina, Japan and Chile sealed passage to the last eight, and after being given an almighty scare by Samoa in their final pool match, England overcame a
Fijian fightback in Marseille to triumph 30-24. Despite being outscored by three tries to two, captain Owen Farrell kicked 20 points, sending his side into the
last four at the expense of 2023's breakout stars.
</item 92>

<item 93>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
New Zealand played some scintillating rugby in victory (AFP via Getty Images)

They led 13-0 and though they may not have realised, no team had ever squandered a lead that large in a World Cup quarter-final. Not that they would have
expected an easy ride from there.

They conceded a penalty straight from the restart as their kick chase was illegally impeded and Sexton duly slotted the three.

Ireland had made something of a statement during the Haka as their fans drowned it out with a rousing rendition of Fields of Athenry and the players formed a
figure of eight in memory of ex-international and Munster coach Anthony Foley, who tragically died in 2016, as they stared down their opponents.
</item 93>

<item 94>
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
</item 94>

<item 95>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
And then there were those 37 phases… the most painful finish in Irish rugby history.

Ireland were left bereft after an agonising defeat (PA Wire)

Instead of being feted as the heroes to finally overcome that insurmountable obstacle, the Ireland players will still have to answer endless questions about the
heavy burden of previous knockout failures and fans will endure at least four more years of choking jokes at their expense.

As for the All Blacks, they looked lost at times during this World Cup cycle and it’s still faintly remarkable that head coach Ian Foster survived last summer
as the home defeats to Ireland, South Africa and Argentina piled up.

Yet somehow, he hung on and turned things round. He has the unwanted title of being the first New Zealand coach to lose a World Cup pool match but now a
semi-final awaits next Friday evening against an Argentina side that, frankly, are a tier below them. Forget the hand-wringing ahead of the tournament, the All
Blacks are one very small step away from another World Cup final.
</item 95>

<item 96>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Brodie Retallick will come off the bench.

Elsewhere, Mark Telea has been recalled ahead of Leicester Fainga’anuku. The tryscoring machine was axed for the quarter-final after missing a team curfew.

“That’s the team we think best for this week,” Foster said.

“Mark has done his time. He made a mistake and accepted what was happening but you don’t linger in that space. He’s been our form winger. We have got a faith in
him and think he is in good place to play this game.

Advertisement

“We really enjoyed Leicester Fainga’anuku]’s game last week and he should be proud of that and its’s a tough selection. It’s a chance for us to get Mark on the
park.”

The All Blacks are 18-point favourites for the opening semi-final.
</item 96>

<item 97>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
The All Blacks eventually prevailed in Paris (Reuters)

The second half followed a similar, rollercoaster pattern as every time Ireland got close, their opponents found a way to stretch the lead. Mo’unga’s sumptuous
dummy and sleight of hand from first-phase lineout ball froze two Irish defenders and led to a scything break before he fed speedster Will Jordan to race over
the whitewash.

That extended the gap to eight points but a penalty try as an Irish driving maul was illegally hauled down closed it to 25-24 with 17 minutes to go and sent
Codie Taylor for 10 minutes in the sin-bin.

Yet this All Blacks team suddenly have a resilience about them and brushed off the underdog status they had been given in the build-up to this game as their
tireless back row continually disrupted breakdowns and a disciplined defence refused to yield.

Jordie Barrett added a penalty to take the score to 28-24 and that’s how it eventually stayed.
</item 97>

<item 98>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
I. Ramirez (R. Rojas, 46'), 15. G. Gomez, 5. Balbuena, 6. Alonso (Villasanti, 76'), 4. Espinoza — 20. R Sanchez, 8. Campuzano (Ojeda, 70'), 10. Almiron, 19.
Sosa — 21. Bareiro (Sanabria, 70').

Argentina vs Paraguay live updates, highlights, commentary

FULLTIME: Argentina 1-0 Paraguay

Argentina are most certainly deserved victors as they walk away with all three points, but they made it harder than it had to be by failing to finish a whole
host of great chances. The entire team was impressive until it came time to shoot, and even Lionel Messi couldn't change the narrative as he hit the post twice.
</item 98>

<item 99>
Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves (The Roar | Sports Writers Blog,
2023-10-26T22:19:21+00:00)
The pain of the days and weeks after that dark July, as well as the bumps and bruises along the way, including as recently as August when they suffered their
biggest defeat to the Boks to lose the inaugural and prestigious Qatar Cup, still runs deep.

“It was quite special to hear some of the things those certain players said from the heart. You know, you think you know someone but once they open up and
really be vulnerable in front of a team, it was quite special to hear,” back-rower Dalton Papali’i said this week.



“It was truly something special when you hear those boys open up. You know those leaders have been through a lot in their careers, the highs and the lows, the
off-field and on-field stuff, and for them to open up, it sort of makes you feel welcomed and you want to play this game for them.
</item 99>

<item 100>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The 2019 Rugby World Cup final between England and South Africa was one of many firsts: the game itself capped off a sensational tournament played outside of
the sport's traditional strongholds for the first time; it was the first final in which South Africa scored a try, and by full-time, the 32-12 score in the
Springboks' favour sealed a record-equalling third title for the Rainbow Nation.

Fast forward four years, and South Africa once again prepare to take on England in an all-or-nothing match to decide which nation gets a shot at World Cup glory
— but this time around, the pair meet in the semi-finals of the 2023 tournament, rather than the decider.

That won't make an ounce of difference for either camp, though, as these two long-time rivals prepare for a battle that could re-define the definition of a
'physical encounter.'
</item 100>
=== END ITEM id=c5b8fa9676990658 ===
