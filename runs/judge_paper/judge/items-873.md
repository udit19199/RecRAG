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

=== ITEM id=03490e4c4be0c23a ===
Gold sentences and facts:
Sentence 0: His career has only blossomed in the years since he took that trip, and while some artists might feel trepidation at the attainment of so much success from such an early age, Mitchell is clear-eyed.
  Fact 0.0: His career has only blossomed in the years since he took that trip.
  Fact 0.1: Some artists might feel trepidation about attaining so much success from such an early age.
  Fact 0.2: Mitchell is clear-eyed.
Sentence 1: Since then he’s been linked to club jobs in Europe but opted to stick around with the national team.
  Fact 1.0: Since then, he has been linked to club jobs in Europe.
  Fact 1.1: He opted to stick around with the national team.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Loading “We just want to win. We come into the club every day and get better, and we just set ourselves up, so when we come to game day, we know we’ve done
everything, to get the result we want.” Maynard had a tumultuous start to September after his infamous collision with Melbourne’s Angus Brayshaw left the Demon
concussed and him defending himself at the tribunal. With a three-match ban hanging over his head, Maynard feared his dream to achieve premiership glory could
be lost before he beat the charge.

Collingwood’s triumph helped banish the pain of their grand final defeat to West Coast in 2018, when Maynard was unfortunate not to receive a free kick for a
block against Willie Rioli in the same passage that led to Dom Sheed’s match-winning goal. “Everyone knows that was a bizarre and crazy sort of week, mentally
[after the Brayshaw incident],” Maynard said. “I wasn’t going well because of the what-ifs, if I missed this.
</item 2>

<item 3>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was also impressive for the Murray Bushrangers in the Coates Talent League, averaging 16 disposals and 3.7 tackles across 10 matches and booted nine goals. His
high natural leap and pace were both prominent at the national Draft Combine with his standing vertical leap of 78cm ranked second overall and his time of 2.96
seconds on the 20m sprint ranked in the top ten.

Weapon: Elusiveness and creativity

AFL player comparison: Tom Papley (Sydney Swans)

Will Graham

State: Queensland

State League Club: Gold Coast Suns Academy

Community Club: Palm Beach Currumbin

Date of Birth: 27/07/05

Height: 186cm

Adapted seamlessly to playing in defence for the Allies after predominantly playing midfield for the Gold Coast Suns Academy.
</item 3>

<item 4>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
"Last week 45,000 fans (versus Portugal), last night (60,000 in Lyon). Rugby's such a small aspect in terms of Australians but our fans have turned up here,
we've seen them and spoken to them and their unconditional love for us, the fact that they came 20,000km to get here and support us, has been awesome.

"My message is just to understand that we're working extremely hard, there's no stone unturned, especially the youth in this team, to leave the jersey in a
better place and just to stick with us.

"I know the results haven't been coming our way the last couple of years but when it does turn, that's what we're building towards.﻿"
</item 4>

<item 5>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
His appointment reflects RA’s desire to go down an administrative path rather than former coach of any standing like Millard would have offered.

Horne, who joined World Rugby in 2015 as a general manager of high-performance before moving into his current role, will sit at the top of the high-performance
tree, with the next Wallabies coach to report into him.

Advertisement

During his tenure with World Rugby, the Sydneysider has played an important role in setting up competitions, high-performance structures and pathways across
developing countries.

He also played a role in helping Super Rugby integrate Fiji Drua and Moana Pasifika into the competition.

Those successes played out in the recent Rugby World Cup, with Chili, Fiji, Georgia and Portugal delivering encouraging performances that positively reflected
World Rugby’s desire to spread and build the game. Horne briefly worked at Saracens as a general manger.
</item 5>

<item 6>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Uses his dazzling speed to break the lines and kicks the ball with pinpoint accuracy on his preferred left foot. Averaged 32.8 disposals and 5.8 clearances
across four matches for the Allies at the National Championships to earn All Australian honours. Was equally dominant for the Tasmania Devils in the Coates
Talent League having more than 30 disposals in seven of his nine matches and also kicked 12 goals. Showed he could match it against more experienced opponents
by having 20 disposals and three clearances in Australia’s win over Carlton’s VFL team.
</item 6>

<item 7>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Already a top tier European owner of OGC Nice, there has been some success, but nothing compared to their French riviera neighbours, where Ratcliffe appears set
to look for inspiration.

Paul Mitchell recently left the principality of Monaco, but he could well be back there in meetings with Ratcliffe who is planning to appoint a sporting
director.

Mitchell fulfilled that role at Monaco after previous stints at Leipzig, Tottenham, Southampton and MK Dons, becoming one of football’s most respected
executives.

A former player in England’s lower leagues, Mitchell was previously linked with Chelsea and Liverpool, and much like Ratcliffe, he’s also from Greater
Manchester.

Starring mainly for Wigan and MK Dons, it was with the latter where Mitchell was moved upstairs, and made his first big move by spotting future England regular
Dele Alli after being named their head of recruitment in 2010.
</item 7>

<item 8>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
However, names such as Alli, Heung-Min Son, Toby Alderweireld, Eric Dier and Kieran Trippier set the side up for a bright future, before chairman Daniel Levy
got in Mitchell’s way.

Leaving what he called his ‘dream job’ after less than two years, it was reportedly due to Levy’s over-involvement in negotiations and the recruitment process,
something which wasn’t an issue at his next club – Leipzig.

Ralf Rangnick landed Mitchell and more success followed. Signings such as Dani Olmo, Christopher Nkunku and Amadou Haidara again showcased what Mitchell can do
with the shackles off, and again the results followed.

6 Getting Nkunku from PSG was one of Mitchell's biggest coups Credit: Getty

Many of his signings starred in a run to the Champions League semi-finals in 2020, and in 2022 won the club’s first ever trophy with a DFB Pokal.
</item 8>

<item 9>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Averaged 17.1 disposals at 76 percent efficiency and kicked seven goals across 12 matches in the Coates Talent League to announce himself as a Draft prospect.
His 6:24 in the 2km time trial at the state Draft Combine in Victoria was impressive for a player of his type.

Weapon: Rebound and ball use

AFL player comparison: Bailey Dale (Western Bulldogs)

Mitchell Edwards

State: Western Australia

State League Club: Peel Thunder

Community Club: South Mandurah

Date of Birth: 02/06/05

Height: 206cm

Advertisement

An exciting ruckman still growing into his body. Has the height and natural leap to give his team a distinct advantage at stoppages. Has continued to develop
this year and shown glimpses of his aerial ability at state under-18 level and took several exciting marks during the National Championships.
</item 9>

<item 10>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
He has been part of the Sydney Swans’ Academy since the under-10s.

Weapon: Contested ball-winning ability

AFL player comparison: Tim Taranto (Richmond)

Lance Collard

State: Western Australia

State League Club: Subiaco

Community Club: North Beach

Date of Birth: 26/03/05

Height: 180cm

Small forward who is extremely dangerous around goals. Booted 39 goals in 16 matches in Western Australia’s pathway this year where he played predominantly at
Colts level. Also represented his state in three games at the National Championships which gave him a taste of playing against the nation’s best. He then
highlighted his potential by kicking five-goal bags in three of the final five games of the WAFL Colts season.
</item 10>

<item 11>
How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
Tyler Mitchell Albany, Georgia, 2021 (detail). Credit: Courtesy of the artist and Jack Shainman Gallery, New York As a 20-year-old film student at New York
University, Mitchell became enamoured with still photography after he signed up to do a documentary photography course in Havana, Cuba. Frustrated by the amount
of time and money required to make films, he was drawn to the accessibility of still photography.

Advertisement

“I was impatient when I got started and just wanted to make visual things now, and I didn’t want to wait for permission, or didn’t want to wait on equipment, or
didn’t want to lug large amounts of equipment up flights of stairs. I just wanted to make something that I felt compelled by, immediately,” he says. His career
has only blossomed in the years since he took that trip, and while some artists might feel trepidation at the attainment of so much success from such an early
age, Mitchell is clear-eyed. “The idea of the road ahead only excites me,” he says.
</item 11>

<item 12>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
A competitive type who has a good natural leap and is very strong overhead. He had an excellent year at under-18 level for Central District where he averaged
21.8 disposals and 5.9 marks across nine matches. He was also one of South Australia’s better players at the National Championships. Is eligible to join
Hawthorn under the father-son rule as the son of Luke McCabe who played 138 games for the Hawks from 1995 to 2004.

Weapon: Overhead marking

AFL player comparison: Jordan Ridley (Essendon)

Advertisement

Colby McKercher

State: Tasmania

State League Club: Tasmania Devils

Community Club: Launceston

Date of Birth: 12/04/05

Height: 180cm

Had an outstanding year which included winning the Coates Talent League’s Morrish Medal and finishing runner-up in the Larke Medal at the National
Championships.
</item 12>

<item 13>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
Arnold – who also represented the Socceroos 56 times as a player – has been involved in national team coaching ranks since 2000, first as an assistant to Farina
and then Guus Hiddink, before a brief period as caretaker coach in 2006.

“It’s certainly a proud moment and an achievement that I’ll reflect on when the time is right,” Arnold said.

“I look at some of the names on that list – Farina, Arok, Thompson, Rasic – guys I played under or assisted in the early part of my coaching career, and it
gives you some perspective on what a journey it’s been so far.

Advertisement

“I think back to that caretaker period and how much I learnt about coaching; it was a challenging experience and one that on reflection I probably wasn’t ready
for. I’m grateful for the opportunity I was given and what it was able to teach me ahead of my pathway into club coaching.
</item 13>

<item 14>
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
</item 14>

<item 15>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Also showed he can go forward and have an impact during his time with the Murray Bushrangers.

Weapon: Rebound ability and athleticism

AFL player comparison: Sam Taylor (GWS Giants)

Ethan Read

State: Queensland

State League Club: Gold Coast Suns Academy

Community Club: Palm Beach Currumbin

Date of Birth 07/07/05

Height: 202cm

Mobile and skillful ruckman whose work rate and ball use are outstanding features of his game, along with his aggressive work at stoppages. A member of the Gold
Coast Suns Academy, he was a dominant force for the Allies at the National Championships and earned All Australian honours. Had a massive impact on all four
games, averaging 21.5 disposals at 76 percent efficiency, 7.5 marks and 13 hitouts to play an important role in the Allies’ first title win.
</item 15>

<item 16>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Consistent high-level performances

AFL player comparison: Andrew Brayshaw (Fremantle)

Ari Schoenmaker

Pronunciation: Show-en-maker

State: Tasmania

State League Club: Tasmania Devils

Community Club: Launceston

Date of Birth: 07/01/05

Height: 194cm

Left-footed tall defender who reads the game exceptionally well, is a good interceptor and prodigious kick of the ball. Had a very impressive back half of the
year, which included being selected for the Allies in their final two matches of the National Championships. He also showed the ability to play on the wing and
averaged 27 disposals at 75 percent efficiency as well as 5.9 marks and 9.6 rebound 50s in his eight matches with the Tasmania Devils in the Coates Talent
League to be one of their most impressive performers.
</item 16>

<item 17>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was also named captain of the All Australian team after leading the Allies to their first title. Caught the eye with a strong performance for the Australia
under-18 team in its representative match against Port Adelaide’s SANFL side and backed it up in the second match against Carlton’s VFL side where he collected
24 disposals to be one of Australia’s best in victory. Finished the year as a back-to-back Coates Talent League premiership player with the Sandringham Dragons
where he averaged 31 disposals across nine matches.
</item 17>

<item 18>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Rugby Australia is moving closer to naming their new director of high-performance, with World Rugby’s director of high-performance Peter Horne set to take on
the role – and the move could help land highly-regarded coach Joe Schmidt for the vacant Wallabies job.

As The Roar reported last week, Horne had emerged as the leading candidate in recent weeks.

It’s believed he has edged out former Sydney University, Australian Sevens and Harlequins’ director of rugby Billy Millard for the role.

Facebook Twitter Whatsapp Reddit Email Share

The finer points of the negotiations are being out, but it’s likely Horne will be announced over the next week once the sports administrator wraps up a crucial
series of meetings this week with World Rugby.

Horne is leading World Rugby meetings with tier-two nations this week in Sydney, with former Fiji coach Simon Raiwalui in attendance before he officially wraps
up and joins World Rugby.
</item 18>

<item 19>
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
</item 19>

<item 20>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Only leaving the side to reach the pinnacle of his job, finally stepping up from a head of recruitment to become sporting director at Monaco, Mitchell showed
just how big picture he is with an interview in 2021.

“I want to continue to drive standards in each and every department,” he said. “Psychology, sports science, nutrition, medicine, the way we travel, the way we
refuel.

“Everything, every element I think we need to keep pushing to its maximum to try and be the best we possibly can.”

Having worked in football management for over a decade, Mitchell’s career has always been about the long term for his club’s, with the young signings he’s made
often only paying off after he’s left.

One in particular was 2022/23 Premier League top scorer Son, who perfectly sums up his philosophy.

“I took quite a lot of criticism in my first year at Tottenham for some of the decisions and Sonny was kind of at the centre of that,” he explained.
</item 20>

<item 21>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Played all four games for Victoria Metro at the National Championships and was his team’s best player in the loss to the Allies, while also shining with 22
disposals in the win over Western Australia. Across his 16 matches with the Eastern Ranges in the Coates Talent League he averaged 19.4 disposals to be one of
their most consistent players during their run to the grand final. Recorded a slick time of 2.92 seconds on the 20m sprint during the national Draft Combine
highlighting his dazzling acceleration and backed it up by recording 6:28 for the 2km time trial which demonstrated a combination of elite speed and endurance
which are essential to play on the wing.

Weapon: Elite speed and ball use

AFL player comparison: Nick Martin (Essendon)
</item 21>

<item 22>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
He got to answer critics like Mark Bosnich and Craig Foster as well – bluntly rejecting their negativity with the dismissive “who cares?”. That would have
tasted sweet.

On Tuesday, FA CEO James Johnson, who held firm on Arnold 20 months ago, got to congratulate the coach on his impending record-breaking match against Bangladesh
as 2026 World Cup qualifying starts on Thursday.

Advertisement

“Graham is one of the greatest contributors to the Socceroos,” Johnson said in a statement.

“He’s been a constant in the Australian football landscape for more than 30 years as both a player and coach, being a part of some of the Socceroos most
memorable moments.

“Anyone who knows Arnie will tell you he bleeds green and gold. His love for the Socceroos and football in Australia is both infectious and inspiring.
</item 22>

<item 23>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Has already played nine matches at senior level with Peel Thunder averaging 18.9 disposals. Starred when he represented Western Australia at the National
Championships averaging 27.2 disposals, 11 contested possessions, six clearances and five tackles to win All Australian honours. Is the son of former West Coast
Eagles and Geelong Cats forward Derek Hall but is not eligible as a father-son to either club given Derek played only two games for the Eagles and 74 matches
for the Cats. Didn’t test at the national Draft Combine due to injury.
</item 23>

<item 24>
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
</item 24>

<item 25>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Continued that form into this season and was outstanding for Western Australia at the National Championships winning his team’s MVP award and All-Australian
honours after averaging 23.5 disposals and 6.5 marks playing as a defender and midfielder. Played a handful of senior games for Claremont and ended the season
being one of his team’s best in a premiership win at under-18 level. His agility, speed for his size and precise kicking are all features of his game.
</item 25>

<item 26>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Versatility and contested ball-winning ability

AFL player comparison: Callum Mills (Sydney Swans)

"All through my junior football, my dad placed a big emphasis on playing every position. I think that came to the fore this year a little bit."



Tassie’s 'Mr Fix-It' James Leake is shaping as a first-round pick following an impressive season.https://t.co/fwnC96Uzdw — AFL (@AFL) November 3, 2023

Kane McAuliffe

State: South Australia

State League Club: North Adelaide

Community Club: Central Augusta

Date of Birth: 01/03/05

Height: 187cm

A powerful midfielder who wins his own ball and is a penetrating left-foot kick. Is a strong and aggressive tackler as well as being an accomplished extractor
of the ball from stoppages using his clean hands and vision to set up the play.
</item 26>

<item 27>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was an outstanding player for Victoria Country at the National Championships to earn All Australian honours, which included a match-winning performance against
Victoria Metro in the final game where he amassed 22 disposals, 10 marks and four goals. Was also dominant for the Gippsland Power to win their best and fairest
and earn selection in the Coates Talent League Team of the Year. Finished the year with 42 goals across 20 matches in all competitions. His natural leap was
highlighted at the national Draft Combine with his running vertical jump of 97cm ranked second overall. Hails from a talented football family, with his brother
Xavier now at Essendon and sister Yasmin playing for Port Adelaide.
</item 27>

<item 28>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Clean hands and vision

AFL player comparison: Hugh McCluggage (Brisbane Lions)

Taylor Goad

State: South Australia

State League Club: South Adelaide

Community Club: Willunga

Date of Birth: 28/04/05

Height: 206cm

Developing ruckman who only began to focus on a potential AFL career this year after initially applying his natural talent to basketball. Showed splendid
improvement throughout the season playing at under-18 level for his state and club. Was particularly impressive in South Adelaide’s premiership campaign where
he averaged 10.4 disposals, 25 hitouts and 2.2 marks in 13 matches. Both his standing vertical jump of 74cm and 20m sprint time of 2.965 seconds were ranked in
the top 10 at the national Draft Combine and gave distinct indications of his elite athletic tools.
</item 28>

<item 29>
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
</item 29>

<item 30>
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
</item 30>

<item 31>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
“Sometimes players need time, they’re human beings, they need to settle, Sonny had a great track record at Hamburg and Leverkusen but there was this narrative
that started to grow internally and externally that he wasn’t a good finisher.

6 Mitchell is becoming an increasingly recognisable face

“I remember telling a coach ‘you’re wrong, his history shows in two different clubs he’s a very astute finisher off both sides’.”

Mitchell remains good friends with Rangnick from their time together at Leipzig, and last season made headlines when he was spotted at a youth team game while
the German was interim coach at Man United.

So astute is Mitchell in his observations, he even laid out a blueprint for United's success while at Monaco.

“I think context is important to measure what Manchester United's current situation is, but I think they do need to get to the point of really putting down what
Manchester United needs to be in the modern game,” he explained.

"We are a long time from the Sir Alex Ferguson era.
</item 31>

<item 32>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
He also impressed for the Australia under-18 team in its second match against Carlton’s VFL side at Marvel Stadium, gathering 26 disposals and six clearances to
claim the medal as his team’s best player.

Weapon: One-touch hands and creativity

AFL player comparison: Touk Miller (Gold Coast Suns)

Advertisement

Ryley Sanders

State: Tasmania

State League Club: Sandringham Dragons

Community Club: North Launceston

Date of Birth: 21/01/05

Height: 185cm

Strong-bodied midfielder who is powerful and courageous in his attack on the ball and the pressure he exerts on the opposition. Strong two-way running is
another feature of his game and was on display in dominating for the Allies at the National Championships, averaging 35 disposals at 80 percent efficiency, 14.5
contested possessions, five clearances and four tackles to win the Larke Medal as the best and fairest player.
</item 32>

<item 33>
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
</item 33>

<item 34>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
We need to maximize the development of the players. At the end of the day, that’s the key to success and alignment of your resources, for a small country.

“Because Australia, if you look at the history of rugby, when we were amateur, we were probably ahead of the game because we had the influence of rugby league.
We had the influence of AFL and our players were more professional than the rest of the world.”

He added: “I know what’s going on and we need to get a better system of developing players. We need to develop better players and we need to develop a larger
batch of better players and that only comes through changing the whole system.”

While all five Australian Super Rugby sides fundamentally are on board with regards to moving to a centralised high-performance program, The Roar added that
some states, including the ACT Brumbies as well as the QRU, don’t want to give up their commercial assets.
</item 34>

<item 35>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
An excellent decision-maker and ball user by hand or foot, he averaged 19.5 disposals, 12.5 contested possessions and 6.5 clearances to finish third in the
Larke Medal voting and took out Victoria Country’s MVP. Further added to his very impressive record by earning selection in the All Australian team for the
second year in a row. Handled the step up to playing senior football with three impressive games in the VFL and was the best player for the Australia under-18
team in its match against Port Adelaide’s SANFL team.
</item 35>

<item 36>
How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
Tyler Mitchell: “I’ve never been to Australia so it’s very exciting to sort of have a global or new conversation on a new continent with my work.″⁣ Credit:
Courtesy of the artist and Jack Shainman Gallery, New York The theme of the Triennial is “Magic, Matter, Memory”, making Mitchell’s pieces a logical inclusion.
“A lot of my work meditates on being between a heightened reality, or potentially even a magical reality and an actual reality,” he says.

Advertisement

“I’m creating what I call lightly staged moments. And so that means I’m setting up scenarios where these families are enjoying real moments of leisure in
outdoor spaces, usually open pastures and fields, and those moments are meant to stand in for a meditation on what it means to enjoy leisurely outdoor nature as
a black person.” Tyler Mitchell, Vastness, 2021.
</item 36>

<item 37>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
With moments left in extra time against the South Americans, Arnold dragged Mat Ryan and replaced him with Andrew Redmayne – who saved the decisive spot kick
(and also threw his opponents water bottle with cheat sheet attached away from the goal).

Advertisement

Ryan might well have gotten the Socceroos to Qatar, but if Australia had not advanced there’s little chance Arnold would still be their coach today.

And there’s virtually no chance he’ll have to endure a similar level of stress through to the next finals – with FIFA expanding the tournament from 36 to 48
teams meaning eight direct Asian confederation qualifiers will proceed directly instead of four.

Arnold, in his second stint as national team manager, will prowl the technical area for a 59th time on Thursday, taking him clear of Frank Farina.
</item 37>

<item 38>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Advertisement

“What’s changed back there that I could think that, what I can offer, can change what they’re doing still?

“I’ll always love rugby. I’ll always love Australian rugby. Could I see myself working in Australia again? I doubt it.”

But RA have been making moves to try and twist Nucifora’s arm.

While Nucifora was never in the running for the director of high-performance, it’s understood they are trying to get him on board as a consultant when he wraps
up with Ireland.
</item 38>

<item 39>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
“I’m super grateful that I’ve got a little chant like that, and hopefully, it’s a little legacy I can continue, and I’m just happy the Magpie army are looking
after me. I feel like they’re always on my side, no matter what mistake I make ... if anything, they motivate me to continue to take the game on and just be my
true self.” Markov felt like he was cheating on Collingwood when he began training with the Blues, after finding out the Pies overlooked him for what was then
their final list spot.

That followed Gold Coast delisting the 27-year-old, who also played 23 games for Richmond. There was interest in him at the time, but he feared his career might
be over. “I went through the national draft and no one looked into me, so there were technically 17 clubs that didn’t really want me,” he said.
</item 39>

<item 40>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
So far RA have said they intend backing Jones and he said he believed he had official support. He refused to answer a question on potential break clauses in his
contract with RA.

Advertisement

On whether he will see out his contract through the end of 2027, Jones said: “It’s not absolutely my decision. “We’ve got a review going forward and let’s see
what happens at the end of the review.

He said he was not worried about being sacked but accepted it was possible.

“I don’t control those sort of things. All I can do is coach. I went to the World Cup, came in for a short period of time and had to make a decision on the
team, made the decision we needed to go with youth, and whilst the results in the World Cup weren’t the results we wanted, I think I’ve left the Australian team
in a great position to go on to 2027.

“That judgment will be decided by the 2027 World Cup.”
</item 40>

<item 41>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Horne briefly worked at Saracens as a general manger.

RA always said they wanted to have the role filled by Christmas, with chief-executive Phil Waugh saying the successful candidate had one of the biggest jobs to
change the game’s standing in Australia.

“It’s a big job,” Waugh said. “It’s the biggest job on my leadership team in terms of the impact it has on the general sentiment around the game.”

Horne will oversee all of Australia’s men’s and women’s high-performance programs.

Advertisement

He will also be tasked with “integrating” the programs, with RA trying to centralise all of the Super Rugby’s high-performance programs.

Perhaps most pressing will be building a list of coaching candidates to take over from Eddie Jones, with RA wanting to have the new Wallabies coach set in stone
by the start of Super Rugby.
</item 41>

<item 42>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
His love for the Socceroos and football in Australia is both infectious and inspiring. That energy and passion is a huge element to the way he’s able to lead
and inspire our nation’s best male footballers in some of our sport’s most challenging arenas.

“On behalf of Football Australia, I’d like to congratulate Arnie on everything he’s achieved in reaching this milestone and we look forward to him continuing to
build on what is already an exceptional legacy.”

In a remarkable few weeks last year Arnold went from on the nose to a man revived. Since then he’s been linked to club jobs in Europe but opted to stick around
with the national team.

While national team coaches make important calls every game, Arnold’s record-breaking achievement is conceivably down to one big ballsy call in the playoff
against Peru.
</item 42>

<item 43>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was impossible to contain during this year’s National Championships booting 14 majors in four matches for Victoria Metro to top the goal kicking. He also
averaged 12.2 disposals and was rewarded with All Australian selection. He further expanded his game by spending more in the midfield and across half back for
the Eastern Ranges and averaged 20 disposals while kicking 11 goals in nine matches in the Coates Talent League.

Weapon: Speed, agility and leap

AFL player comparison: Shai Bolton (Richmond)

Advertisement

This is just what Nick Watson does ????
</item 43>

<item 44>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Adams and McStay – who arrived from the Lions in the off-season as an unrestricted free agent – leaned on each other to cope with the difficult moment, and
needed some privacy to deal with their personal misfortune before they could share in the team’s raucous celebrations. McStay may need as many as eight weeks to
recover from the high-grade medial collateral ligament strain he suffered in the club’s preliminary final win. Collingwood’s injured recruit Dan McStay at
training before the grand final. Credit: Eddie Jim “We shed a few tears and a bit of emotion, then we brushed it off and said, ‘Let’s get on with it and go
celebrate with the boys’. It was a great night,” Adams said. John Noble was another sad story, playing every game in the home-and-away season, only to be
dropped for Collingwood’s qualifying final clash with Melbourne and never forcing his way back in.
</item 44>

<item 45>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
He only fully focused on a potential AFL career last year after previously playing volleyball. Starred with the Calder Cannons and Victoria Metro this year
kicking 33 goals in the talent pathway. His 2km time of 6:05 at the national Draft Combine was a strong indicator of his elite endurance but also his
competitiveness for a player of his size. Is eligible to join the Western Bulldogs under the father-son rule as the son of former Bulldogs utility Matthew
Croft, who played 186 games for the club.

Weapon: Aerial ability

AFL player comparison: Max King (St Kilda)

Jordan Croft has nominated as a father-son prospect for the Western Bulldogs ????
</item 45>

<item 46>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
His 6:37 for the 2km time trial at the national Draft Combine was a strong performance for a player of his type.

Weapon: Contested ball-winning and ball use

AFL player comparison: Tom Green (GWS Giants)

Koltyn Tholstrup

Pronunciation: Foal-strup

State: Western Australia

State League Club: Subiaco

Community Club: Newtown Condingup

Date of Birth: 28/06/05

Height: 186cm

A talented player with an elite work rate who often plays as a high half forward and sometimes goes through the midfield. He is a fearless competitor with
excellent game sense and a penetrating kick. He played the final three games for Western Australia at the National Championships after missing the first with
injury and didn’t miss a beat to average 18.7 disposals.
</item 46>

<item 47>
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
</item 47>

<item 48>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
NEW PODCAST! Michael Atkinson and Sam Worthington preview a thrilling final round of pool games to decide the quarter-finalists at the Rugby World Cup in
France﻿

He wanted to regain form at club level ﻿with Urayasu D-Rocks but his hand remained up for Test selection.

"I chose to leave Australian shores and understand the ruling around that and what it holds," he said.

"I've got to perform for my club first and if you get the call up, you get the call up. If you don't, I'll take my three months break and enjoy that."

Kerevi is an unabashed Jones fan.

Eddie Jones with fans during a Wallabies training session at Stade Roger Baudras. (Getty)

"He's up at 4am working hard. We catch him in the gym, he's working hard and I swear he doesn't even sleep.
</item 48>

<item 49>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
His ferocious attack on the ball and his opponent helped set the tone for the Allies to dominate the midfield battle throughout the National Championships and
he was rewarded with All Australian honours. Averaged 29.2 disposals, 11.2 contested possessions and a team-high 6.8 tackles across four matches for the Allies
and showed excellent repeat speed and courage as well as clean hands playing both on the inside and outside. His 6:06 effort in the 2km time trial at the
national Draft Combine showcased his elite endurance. He has been part of the Sydney Swans’ Academy since the under-10s.
</item 49>

<item 50>
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
</item 50>
=== END ITEM id=03490e4c4be0c23a ===

=== ITEM id=297249d4e8bebce1 ===
Gold sentences and facts:
Sentence 0: Tyreek Hill had two touchdowns among his 157 receiving yards to help the Miami Dolphins rout the Washington Commanders (4-9).
  Fact 0.0: Tyreek Hill had two touchdowns in this game.
  Fact 0.1: Tyreek Hill had 157 receiving yards in this game.
  Fact 0.2: Tyreek Hill helped the Miami Dolphins rout the Washington Commanders.
  Fact 0.3: The Washington Commanders were 4-9 at this point.
Sentence 1: • Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games.
  Fact 1.0: Tyreek Hill made six catches for 163 yards and a touchdown against Carolina.
  Fact 1.1: Tyreek Hill had 812 yards through the first six games.
Sentence 2: Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight.
  Fact 2.0: Miami has five remaining opponents.
  Fact 2.1: Four of Miami's five remaining opponents have pass defenses ranked in the league's top eight.

Retrieved items:
<item 1>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The October 1 matchup between the Chiefs and the Jets drew 27 million viewers. And became the most-watched Sunday TV show since the last Super Bowl! And the
last Super Bowl had an actual concert by Rihanna.

Revealed: The Dark Side Of Pat McAfee & Why Rob Gronkowski Is In The Firing Line

Kaepernick, who? Deflategate, what? In the media frenzy surrounding the two lovebirds, other issues get thrown out of the window. How can fans forget that the
NFL is also organizing games outside the American borders with a view to garnering international appeal? The NFL has scheduled 5 games abroad this season. It is
also addressing the concussion controversies. It recently added both new concussion rules and a new $100 million initiative called “Play Smart, Play Safe,” to
make the sport safer for players. And all the effort is showing.
</item 1>

<item 2>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
$28 billion. And just like that, their target will be met.

USA Today via Reuters NFL, American Football – Taylor Swift reacts while sitting next to Donna Kelce watching the Kansas City Chiefs vs Chicago Bears game
during the first half at GEHA Field at Arrowhead Stadium, Kansas City, Missouri, USA | Image Credits: Denny Medley-USA TODAY Sports/Reuters

And while analysts like Stephen A. Smith may go around telling people they should be embarrassed of themselves for getting so invested in someone else’s
relationship, it is good business for the NFL. TMZ reported that #87’s jersey sales skyrocketed 400% and his Instagram followers by 276%. Yahoo Finance reported
a 53% rise in viewership among teen girls. Who would have thought in their wildest dreams that the NFL would manage to rope in this part of the population? The
October 1 matchup between the Chiefs and the Jets drew 27 million viewers.
</item 2>

<item 3>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Who do you want lining up the potential winning kick in a Super Bowl? It’s an interesting question in a season that has seen Moody and Saints rookie Blake Grupe
miss game-winning tries, but the data could be on the 49ers’ side.

FGA Dist Rookies Veterans <33 Yds 97% [31-32] 94% [330-351] 34-44 Yds 75% [21-28] 82% [230-279] 45-50 Yds 78% [7-9] 73% [121-165] 51-55 Yds 100% [8-8] 59%
[33-56] 56+ Yds 0% [0-1] 55% [6-11] Totals 86% [67-78] 84% [720-862]

The table above shows field-goal make rates in the postseason since 2000 for rookie kickers vs. veterans.
</item 3>

<item 4>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Especially when you face a defensive coordinator who says, ‘You show me you can beat me throwing the ball outside the numbers — best of luck, and to do that,
you are going to have to take a five-step drop and I’m going to have Myles Garrett chasing you every play.'”

Advertisement

• Kansas City Chiefs: Is the explosive passing game gone for good?

What I think: The Chiefs have never scored less per game on offense through six games or a full season since Patrick Mahomes became their quarterback. They have
also never enjoyed larger average point differentials (9.8 per game), because their defense is so much better. I think the pass offense will become slightly
more explosive as Mahomes gets reps with his receivers, but I’m not expecting a huge shift.

The Chiefs are averaging 9.5 plays per touchdown drive when starting at least 70 yards from the opponent’s goal line, per TruMedia.
</item 4>

<item 5>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
1 seed in the conference.

Compared to the 49ers, Cowboys and Eagles, Goff and the Lions may have the easiest schedule to finish the season.

But bettors should also pay attention to C.J. Stroud. He leads the league in passing yards, is one of four quarterbacks eho have started the whole season with a
passer rating over 100, and his touchdown to interception ratio is the best in the NFL (4 to 1)

As for the value of his play, Stroud has taken one of the worst teams in the NFL (11 total wins the last three seasons) and turned it into a playoff contender
and potential division champion.

Offensive Player of the Year

Current favorite: Tyreek Hill

Longshot: CeeDee Lamb

There's a good reason why Tyreek Hill is such a heavy favorite to win OPOY. The Dolphins wide receiver is on track to become the first player with 2,000-plus
receiving yards in a single season.
</item 5>

<item 6>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The hardcore football fans are pretty much done with the NFL harping about Travis Kelce and Taylor Swift. Reality TV star Rachel Lindsay, a longtime NFL fan,
recently slammed the “obnoxious” coverage, highlighting the “exhausting” uptick in Swifties obsessing over football. But that’s just one example of how fans are
finding this fixation a little too disrespectful towards the sport.

America’s Favorite Video Today

From the league’s humble beginnings in 1920 to the $18 billion juggernaut that it is today, the ride has been rocky with more than just a few small bumps along
the road. But with the ambitious target that NFL Commissioner, Roger Goodell, has set for 2027, is there a new strategy in the 104-year-old league’s master plan
to dominate the world sporting scene?

Why the ‘Traylor’ saga will be the key to Roger Goodell’s $27 billion target?
</item 6>

<item 7>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
And all the effort is showing.

According to Joe Pompliano, “The NFL will bring in $20 billion in revenue this year, more than the Premier League, La Liga, Bundesliga, Serie A, and Ligue 1
combined.” Coming to the strategy part, a curious mind would perhaps think can they take a leaf out of the NBA’s playbook and reinvigorate its appeal as it aims
to conquer international audiences?

Can the NFL take a leaf out of the NBA’s grisly past and glorious recovery?

The NFL is already the world’s richest league. Calling it an incredible feat would be a gross understatement. Consider this: the English Premier League, which
is broadcast in over 212 territories, has a revenue of just $6 billion, while football games are primarily viewed in America only. However, the NFL does not
just want to stop at $18 billion.
</item 7>

<item 8>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 8>

<item 9>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Six of the 17 won anyway. The Dolphins were the only one of those 223 teams to win and cover the spread, which was 13.5 for this game.

• Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games. That ranks second in NFL
history to Hall of Famer Don Hutson, who had 819 yards through six with Green Bay in 1942.
</item 9>

<item 10>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
They are gaining more than 15 yards on just 11 percent of pass plays, down from 21 percent through six games last season and 19 percent for 2022 overall. Jalen
Hurts, A.J. Brown, DeVonta Smith and Dallas Goedert are still on the team, so I’m inclined to trust the production will return. But there’s also a lingering
fear defenses have caught up to aspects of the Eagles’ scheme. Does Philly have a counterpunch?

Advertisement

Exec comment: “People are going to crush Hurts and the offense. They are just churning the clock so much. It just feels like they are imposing their will on
teams through the run game and through their short passing and it just hasn’t taken off from an explosive passing standpoint, so when they play a team that can
move the ball on them through the air, it presents some challenges.”
</item 10>

<item 11>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson.

-- Deebo Samuel is on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons.
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson. 18 -- Puka Nacua needs 18 more catches to set the rookie record for most catches in a
season.

-- Puka Nacua needs 18 more catches to set the rookie record for most catches in a season. 152.67 -- After missing last week's game, Tyreek Hill now needs to
average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.

-- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving
yards this season.
</item 11>

<item 12>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Receiving Yards Through Six NFL Games Rk Player Season Yds 1 Don Hutson 1942 819 2 Tyreek Hill 2023 812 3 Wes Welker 2011 785 4 Elroy Hirsch 1951 766 5 Santana
Moss 2005 743 6 DeAndre Hopkins 2015 726 7 Raymond Berry 1960 725 8 Jerry Rice 1986 723 9 Bob Boyd 1954 718 10 Bobby Mitchell 1962 714

• Jared Goff is enjoying the second-best start to a season for his career from a statistical standpoint, behind only his 2018 Super Bowl season with the Rams.
With the Lions at 5-1 and Goff heading toward a likely contract extension, could another team come after his injured backup, former University of Tennessee
quarterback Hendon Hooker?
</item 12>

<item 13>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Tyreek Hill (@ Commanders in Week 13) did his thing despite a tough matchup in Week 12, resulting in Miami's WR1 returning to the top spot against a putrid
Washington stop unit. CeeDee Lamb (vs. Seahawks) isn't far behind, as Dallas' WR1 aims to maintain his efficient play against Seattle.

Amon-Ra St. Brown (@ Saints) has been a target hog the past several weeks, seeing at least nine looks in each of his past six games. Don't expect his workload
to decrease in a bounce-back game for the Lions. Keenan Allen (@ Patriots) looks to continue his dominant ways, as Justin Herbert's go-to target routinely has
double-digit target potential.

Mike Evans (vs. Panthers) sits in our top five for Week 13, totaling four TDs over the past three weeks. A.J. Brown (vs.
</item 13>

<item 14>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
GO DEEPER Behind the AI magic that lets Amazon's Prime Vision show the NFL like never before

• Buffalo Bills: Are the defensive injuries too much to overcome?

What I think: The rest of the conference is the key variable. The Bills are diminished. They have imperfections. But Buffalo already defeated the Dolphins. The
other AFC teams aren’t exactly peaking. Kansas City and Cincinnati appear easier to defend this season than in the recent past. We’ll see how well the Dolphins
hold up. This can still be Buffalo’s season.

Exec comment: “Matt Milano and Tre’Davious White, those are huge injuries for that football team. The Bills are not as dominant as they felt going into it with
so many injuries on defense. But the division they are in, they will still be able to slug it out and get to where they need to go. Nobody has really emerged in
the AFC to unseat these teams at the top.”
</item 14>

<item 15>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
This team is relatively young. They are scrappy. The fact that they have been there before gives them a chance. I think the quarterback is growing.”

• Cincinnati Bengals: Can Joe Burrow hold up for a full season?

What I think: Burrow is looking healthier over the past two weeks. The bye week comes at a great time for Burrow to get even healthier, but check out the
schedule on the other side: San Francisco, Buffalo, Houston, Baltimore, Pittsburgh and Jacksonville. Cincy still has a second game with the Steelers, plus games
against Kansas City and Cleveland. I’m worried Burrow will wear down against these defenses.

Exec comment: “You cannot always expect Joe Burrow to create magic with a bottom-half offensive line. The line is my concern all the way.”

Advertisement

2. Are teams sometimes going too far playing to win? The Eagles come to mind after falling 20-14 to the Jets.
</item 15>

<item 16>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Baltimore Ravens: Are the Ravens better off shifting to more of a pass-oriented offense?

What I think: Baltimore has never scored fewer points or amassed less offensive EPA through the first six games of a season with Lamar Jackson in the lineup.
The schedule and injuries have played roles, but with the Ravens shifting toward a more conventional pass offense, are the explosive quarterback runs mostly in
the past? Jackson has one rush longer than 20 yards (he’s had four or five by now previously). He’s gaining at least eight yards on 15 percent of carries, half
his previous rate. I’m not expecting a consistent precision passing game to suddenly materialize. Will this new offense be tougher to defend?

Advertisement

Exec comment: “The defense keeps them in it, but I don’t necessarily think this offensive change is going to make them any better when it counts. Lamar Jackson
does not appear as dynamic as a runner, but in this offense, he does not need to be.
</item 16>

<item 17>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
But there is a solid chance he does not hit that mark. Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight. If he
slows down, someone like CeeDee Lamb could catch up.

Based on his current season average, Lamb is on track for close to 1,600 yards, but based on his production over the last seven games, he could finish with
closer to 1,800 yards.

He will probably need Hill to fade down the stretch to have a shot. But if Hill does and he continues to produce, Lamb could steal OPOY from Hill.

Defensive Player of the Year

Current favorite(s): Micah Parsons , Myles Garrett , and T.J. Watt

, Myles Garrett , and T.J. Watt Longshot: Danielle Hunter

Hunter leads the league in tackles for a loss and is third in sacks with 13.5 (behind Khalil Mack and Watt).
</item 17>

<item 18>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.
</item 18>

<item 19>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Jets would have taken possession with barely more than a minute remaining, no timeouts and Zach Wilson behind center facing a ferocious Eagles pass rush.
Was having Hurts pass in a situation when gaining a first down is unlikely the smart play under the circumstances?

“Are coaches not willing to go against the slight statistical benefit that WPA (win probability added) provides?” the exec asked. “Is it no longer OK to let
your defense win the game when it clearly has a huge talent/performance advantage? This Eagles game was 14-12, not 37-35.”

Critics attuned to the trend toward aggressiveness pounce when teams pass up chances to go for the kill. Coaches know team owners tend to be fans, not football
strategists. They know owners listen to these criticisms. That can provide an incentive to make the aggressive play, to go down swinging.

“Everybody thinks they are playing Brady, Manning or Rodgers in two-minute,” a veteran coach said.
</item 19>

<item 20>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
MIAMI DOLPHINS | WR Tyreek Hill: Arguably the best wide receiver in the NFL, Hill has somehow reached a new level of dominance in his age-29 season. The
seven-time Pro Bowler leads the NFL in receiving yards (1,076) and TD catches (eight). As long as the Dolphins maintain their success on offense down the
stretch, Hill should earn Offensive Player of the Year honors. He's on pace to break Calvin Johnson's season record for receiving yards (1,964).



NEW YORK JETS | LB Quincy Williams: While his brother, Quinnen, may be the better-known Jets defender in the family, Williams — the AFC Defensive Player of the
Month for October — is also making a name for himself. Aside from being the fifth-highest-rated linebacker in the NFL by Pro Football Focus, Williams is tied
for 17th in tackles (74) and tied for ninth in TFLs (nine) among all defenders.
</item 20>
=== END ITEM id=297249d4e8bebce1 ===

=== ITEM id=d450d8f80acc1aa0 ===
Gold sentences and facts:
Sentence 0: Rangers also make a lot of choices during character creation (and leveling up) that determine how they play.
  Fact 0.0: Rangers make a lot of choices during character creation.
  Fact 0.1: Rangers make a lot of choices during leveling up.
  Fact 0.2: The choices Rangers make during character creation and leveling up determine how they play.
Sentence 1: During character creation, you must pick the otherworldy patron that you’ve decided to serve.
  Fact 1.0: During character creation, you must pick an otherworldly patron.
  Fact 1.1: The patron you must pick during character creation is the one you have decided to serve.

Retrieved items:
<item 1>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Finding the best class really just boils down to how you want to play. Even when playing solo, you’ll still manage the rest of your party, so you’ll get to
experiment with multiple classes at once.

BG3 classes have strengths outside of combat

Your character’s class is mostly about combat, but the underlying Ability Score that it relies on affects the other skills you’ll use to interact with the
world. Baldur’s Gate 3 assigns your Ability Scores automatically (though you can tweak it in the character creator), so each class’s primary ability will be
your highest score.

Bards and Sorcerers are both Charisma-based classes, for example. Their high Charisma will also make them good at skills like Persuasion, Deception, and
Intimidation. Out of combat, those classes will be better at talking their way out of (and into) trouble and lying. Wisdom-based classes like Clerics or Druids
will be more insightful and perceptive of their surroundings.
</item 1>

<item 2>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Picking the best class in Baldurs Gate 3 is one of the most important decisions you’ll make in the game — and you have to do it right at the start, with minimal
context to work off.

Your class mostly determines how your character will fight in Baldur’s Gate, but with 12 classes — and 46 subclasses total across them all — it’s not really an
easy decision. Here’s what to know about finding the best class (and the best subclass) for you in Baldur’s Gate 3.

All BG3 classes are ‘good’

The classes in Baldur’s Gate 3 — and in pen-and-paper D&D, for that matter — are all fairly well-balanced. There’s no single best class (barrelmancer aside) in
Baldur’s Gate 3, and the game gives you as many opportunities for creative problem solving as it does for punching. Finding the best class really just boils
down to how you want to play.
</item 2>

<item 3>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
(The Archfey exclusive) Witch Bolt Link yourself to a target with a bolt of lightning. Deal an additional 1-12 Lightning damage each turn by activating it.

Actions

Warlocks have one action, “Fey Presence,” which is exclusive to The Archfey subclass.

Features

Warlock Spell Slot — Used to cast Warlock spells, which can be regained by taking a short rest

Proficiencies

Weapons — Simple Weapons

— Simple Weapons Armor — Light Armor

— Light Armor Saving Throws — Wisdom and Charisma

— Wisdom and Charisma Skills — Choose two skills from Arcana, Deception, History, Intimidation, Investigation, Nature, and Religion

Warlock subclasses

The Warlock has a total of three subclasses, which are unlocked at at the beginning of the game. During character creation, you must pick the otherworldy patron
that you’ve decided to serve.

The Fiend

Warlocks with The Fiend subclass have struck a deal with a hellish fiend who strives for corruption and destruction.
</item 3>

<item 4>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Rangers get Light and Medium Armors and some of the more basic weapons.

Rangers also make a lot of choices during character creation (and leveling up) that determine how they play. At level 1, you’ll choose a Favored Enemy — a
preferred quarry — and a preferred environment that grants you some magical spells.

At level 3, you get to choose a Ranger subclass from Beast Master, Gloom Stalker, or Hunter. Beast Masters get a Companion animal that fights along side you,
Gloom Stalkers focus on stealth, and Hunters get to specialize their melee abilities.

Hunter is the best subclass for a Ranger. Consult our Ranger build guide for more details on why, and to see which skills and feats to pick.

Best class for damage: Barbarian

Barbarians are all about combat. They get light and medium armor (and shields), but they’ve also got the option to keep their AC up even without wearing armor.
</item 4>

<item 5>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
They also get a way to heal themselves and way to get a second attack once per short rest.

At level 3, you get to choose a subclass from Battle Master, Eldritch Knight, and Champion. Battlemasters get a set of Superiority Dice that let you perform
Maneuvers to control the flow of battle and the battlefield. Eldritch Knights add magic into the Fighter’s arsenal. Champions keep it simple with a better
change to deal a Critical Hit (for double damage).

Champion is the best subclass for a Fighter. Consult our Fighter build guide for more details on why, and to see which skills and feats to pick.

Best ranged class for beginners: Ranger

Rangers are agile hunters that focus on Dexterity as their primary Ability. Rangers are an interesting mix of melee, stealth, skills, and magic. That also makes
it harder to get the most out of the class. Rangers get Light and Medium Armors and some of the more basic weapons.
</item 5>

<item 6>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Your second most important stat kind of depends on how you play outside of combat, but Constitution is also a good option for the hit points — Warlocks, like
Sorcerers, have a tendency to be glass cannons. Make Charisma your highest stat and Constitution your second highest.

You’ll pick your Warlock subclass at level 1, but the pact you choose here is actually less important than the Eldritch Invocation you get at level 2 — that
said, go with the Great Old One for the most broadly useful features. Your Eldritch Invocation powers up your Eldritch Blast spell — the bread-and-butter spell
for Warlocks — and turns it into an Agonizing Blast.

Best Warlock feats in BG3

Every four levels, you’ll get the option to either increase your stats or choose a Feat. Feats are special talents that add features to your character. If
you’re happy with your stats and start taking Feats, Warlocks can benefit from:

Actor .
</item 6>

<item 7>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
For more information on the classes in Baldur’s Gate 3, consult our overviews on the Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue,
Sorcerer, Warlock, and Wizard classes.
</item 7>

<item 8>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
They’re the most magic-focused of the melee classes. They get all armors and shields, along with a good mix of weapons. Choosing from the three subclasses —
Oath of the Ancients, Oath of Devotion, and Oath of Vengeance — determines what kind of magic you get. Oath of the Ancients adds in healing magic, Oath of
Devotion gets defense-focused magic, and Oath of Vengeance adds offensive magic. Paladins also get a handful of other spells for dealing out even more holy
justice.

Oath of the Ancients is the best subclass for a Paladin. Consult our Paladin build guide for more details on why, and to see which skills and feats to pick.

Best class for speed: Monk

Monks are a melee class kind of like Fighters, but with a lot more pizzazz. Their best Ability is Dexterity, so their combat is more martial arts than
swordplay.
</item 8>

<item 9>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Best magic class for beginners: Warlock

Warlocks are casters that made a deal with a higher power for their magic. They play a lot like Sorcerers, but with a slight emphasis on more academic Skills
and fewer spells. Warlocks get Light Armor, so they’re a little less squishy than Sorcerers.

Like Sorcerers, you’ll pick your Warlock subclass during character creation. You’ll choose from The Fiend, The Great Old One, and The Archfey subclasses that
are all themed around where your magic comes from — and determine some of the spells you can use. The Fiend Warlocks get their magic from a demon-like fiend and
have hell-themed spells, the Great Old One is more Lovecraftian horror-themed, and the Archfey draws power from a powerful Fey being.

Great Old One is the best subclass for a Sorcerer. Consult our Sorcerer build guide for more details on why, and to see which skills and feats to pick.
</item 9>

<item 10>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Is Warlock the best class for you in BG3?

Warlocks sacrifice the number of spells that they know for sheer power. Every Warlock spell slot is the same — the highest level available — and they recharge
on a short rest instead of a long rest. Warlocks aren’t as versatile as other caster classes, but what they do, they do well.

Warlocks are a great class for you if you want to play a dedicated caster class, but don’t want to deal with all the prep work that classes like Wizards
require. You’ll still get to use (a lot of) magic, but you won't have to manage a binder full of spells every morning.
</item 10>

<item 11>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
There’s a lot of overlap in the classes — especially as you level up. And that means there’s no easy way to say one class is best (or even better). Instead,
focus on what you want to do (or how you want to do it), and go from there.

We’ll list them all below ranked (subjectively) by how difficult they are to play. We’ve put the caster classes toward the more difficult end simply because
they’ve got more to manage with their spells and spell slots — punching things is easier than memorizing spells.

Best melee class for beginners: Fighter

Nothing against Fighters here, but they’re the easiest class to wrap your head around. You’re not going to need to manage spells or Ki Points. Instead, you’ll
see a baddie and then you’ll stab said baddie.

Fighters start out as a very straightforward melee class. With high strength, they get all armors and shields, and choose a method of combat to focus on (and
get buffs to).
</item 11>

<item 12>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Best class for social interactions: Sorcerer

Sorcerers get their magic from within themselves. Sorcerers are the most flexible and varied in the spells they can use. Their primary Ability is Charisma, so
they’re also good at dialogue-heavy encounters out of combat. Sorcerers don’t get any armor, so you’ll have to keep them out of melee combat most of the time.

You’ll pick your Sorcerer subclass when you create your character from the Wild Magic, Draconic Bloodline, and Storm Sorcery subclasses. Wild Magic leans into
the chaotic and untamed nature of magic, Draconic Bloodline gives a dragon-based flavor to your magic, and Storm Sorcery lets you Fly whenever you cast a spell.

Draconic Bloodline is the best subclass for a Sorcerer. Consult our Sorcerer build guide for more details on why, and to see which skills and feats to pick.
</item 12>

<item 13>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Spell Sniper. You learn a cantrip, and the number you need to roll a Critical Hit while attacking is reduced by 1. This effect can stack.

For more information on the classes in Baldur’s Gate 3, consult our overviews on the Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue,
Sorcerer, and Wizard classes. If you’re stumped by the choice, consult our guide on what class to start with.
</item 13>

<item 14>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
The Warlock is one of twelve classes in Baldur’s Gate 3. Warlocks are extreme knowledge-seekers who are willing to form a pact with otherwordly beings in order
to gain ancient secrets that’ll bolster their magical arsenal.

To deal with such powerful creatures, Warlocks need to have strong mental power. They need to be able to make a deal with the devil and survive. As such, their
primary ability score is Charisma.

There’s no single best class in Baldur’s Gate 3, and the game gives you as many opportunities for creative problem solving as it does for punching. Finding the
best class really just boils down to how you want to play. Even when playing solo, you’ll still manage the rest of your party, so you’ll get to experiment with
multiple classes at once.

Read on to find out everything you need to know about the Warlock class in Baldur’s Gate 3, and learn all about the best Warlock subclass, feats, and build.
</item 14>

<item 15>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
If you forge a pact with a fiend, you’ll receive a feature exclusive to The Fiend called “Dark One’s Blessing,” which will grant you four temporary hit points
whenever you defeat a hostile creature.

The Fiend also has two exclusive spells — Burning Hands and Command.

The Great Old One

Those serving The Great Old One are bound to a mysterious being from the Far Realm, who might not even know you exist, The Great Old One will grant you a
feature exclusive to this subclass called “Mortal Reminder,” which will frighten a creature and any nearby enemies after you land a critical hit unless they
succeed a Wisdom saving throw.

The Great Old One also has two exclusive spells — Dissonant Whispers and Tasha’s Hideous Laughter.

The Archfey

If you choose The Archfey as your subclass, your patron is a lord or lady of the fey, a creature from before the time of mortal races.
</item 15>

<item 16>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
For weapons, Barbarians get a wide range of heavy hitters like battleaxes and warhammers. Every Barbarian gets the Rage ability that lets them hit even harder
and reduce damage during combat (and only during combat).

When you hit level 3, you get to choose from the Wildheart, Berserker, or Wild Magic subclasses. Wildheart lets you modify your Rage ability with animal-themed
tweaks like healing, jumping, or a stampede. Berserker is a lot more straightforward and focuses on hitting (extra) things and throwing. And Wild Magic adds a
magical flair to your Rage with a random magical effect and improved saving throws against magic for you and your allies.

Berserker is the best subclass for a Barbarian. Consult our Barbarian build guide for more details on why, and to see which skills and feats to pick.

Best class for magic: Druid

Druids are the nature-based casters of Baldur’s Gate 3.
</item 16>

<item 17>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Most of their magic is nature- or animal-themed and they get Skills to match. What they lack in armor proficiencies — they only get Light and Medium Armor —
they make up for with the ability to shapeshift into a badger, wolf, spider, or cat.

At level 3, Druids pick from the Circle of the Land, Circle of the Moon, and Circle of Spores subclasses. Circle of Land gets you some extra magic and the
ability to cast more spells throughout the day, Circle of the Moon adds a bear to the repertoire of animal shapes you can shift into, and Circle of Spores adds
in some extra fungus- and decay-based magic.

Circle of the Moon is the best subclass for a Druid. Consult our Druid build guide for more details on why, and to see which skills and feats to pick.

Best class for stealth: Rogue

Rogues are Dexterity-based, sneaky fighters. They only get Light Armor, so they have to focus on staying unseen and quick.
</item 17>

<item 18>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Wizards don’t get any armor and are limited to just daggers and quarterstaffs (quarterstaves?) in combat, so they’re going to rely on magic to do any damage.

Your Wizard subclass is chosen when you create the character. You’ll choose from eight schools of magic: Abjuration, Conjuration, Divination, Enchantment,
Evocation, Illusion, Necromancy, and Transmutation. Each subclass gets an extra ability themed to the school of magic. Broadly speaking, Abjuration is about
protection and defense, Conjuration is about creating something from nothing, Divination gets to foresee (and change) rolls, Enchantment is about charming and
controlling minds, Evocation shapes and controls spells, Illusion is about trickery, Necromancy controls life and death, and Transmutation focuses on alchemy
and changing the world around you.

Evocation is the best subclass for a Wizard. Consult our Wizard build guide for more details on why, and to see which skills and feats to pick.
</item 18>

<item 19>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Your Charisma Increases by 1, to a maximum of 20. Your Proficiency Bonus is also doubled for Deception and Performance Checks.

. Your Charisma Increases by 1, to a maximum of 20. Your Proficiency Bonus is also doubled for Deception and Performance Checks. Elemental Adept . Your spells
ignore Resistance to a damage type of your choice. When you cast spells of that type, you cannot roll a 1.

. Your spells ignore Resistance to a damage type of your choice. When you cast spells of that type, you cannot roll a 1. Moderately Armored . You gain Armor
Proficiency with Medium Armor and shields, and your Strength or Dexterity increases by 1, to a maximum of 20.

. You gain Armor Proficiency with Medium Armor and shields, and your Strength or Dexterity increases by 1, to a maximum of 20. Spell Sniper.
</item 19>

<item 20>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Where Bards come into their own, though, is with Bardic Inspiration — a way to boost the rolls of their allies — and their spells that are a nice mix of utility
and offensive magic.

Bards get three subclasses at level 3: College of Lore, College of Valor, and College of Swords. College of Lore Bards get a lot of useful skills to use outside
of combat and get to use their Bardic Inspiration to make enemies weaker, College of Valor Bards help their allies hit even harder, and College of Swords Bards
focus on a swashbuckling style of combat.

College of Valor is the best subclass for a Bard. Consult our Bard build guide for more details on why, and to see which skills and feats to pick.

Best class for magic damage: Wizard

Wizards are the archetypical caster class. They are the only Intelligence-based class in Baldur’s Gate 3 and have the most spells — and the most spell slots to
cast them — of all of the casters.
</item 20>

<item 21>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Each Domain has different spells, actions, and even proficiencies associated with hit. Broadly, Life Domain focuses on healing, Light emphasizes Radiant and
Fire magic (in and out of combat), Trickery is good for sneaking, Knowledge gets extra Skills out of combat and brain-based spells, Nature is flavored a little
more like a Druid, Tempest is a little more combat and Thunder damage-focused, and War is all about combat.

Light Domain is the best subclass for a Cleric. Consult our Cleric build guide for more details on why, and to see which skills and feats to pick.

Best support class: Bard

Bards are a specialized kind of caster class that focus more on performance and charm than a weighty list of spells. Bards only get Light Armor and focus on
Charisma as their primary stat, so they rarely belong on the front line of a fight.
</item 21>

<item 22>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
(The Great Old One exclusive) Expeditious Retreat Gain Dash immediately and as a bonus action on each of your turns until this spell ends. Faerie Fire All
targets within the light turn visible, and Attack Rolls against them have advantage. (The Archfey exclusive) Hellish Rebuke React to your next attacker with
flames that deal 2-20 Fire damage. Hex Make your attacks deal an addition 1-6 Necrotic damage to the target and give it Disadvantage on an Ability of your
choosing. Protection from Evil and Good Protect an ally against the attacks and powers of aberrations, celestials, elementals, fey, fiends, and undead. Tasha's
Hideous Laughter Leave a creature Prone with laughter, without the ability to get up. (The Great Old One exclusive) Sleep Put creatures into a magical slumber.
Select targets up to a combined 24 hit points. (The Archfey exclusive) Witch Bolt Link yourself to a target with a bolt of lightning.
</item 22>

<item 23>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Should you play a custom or an Origin character in BG3?

Baldur’s Gate 3 gives you the option to play as an Origin character, meaning you can just choose one of the NPCs from your party and play as them right out of
the gate. It’s a perfectly fine way to play if you really don’t want to go through the process of picking out a race and background for your character.

That said, you’ll also be limited to that character’s predetermined story. It’s a bit more work at the beginning, but you’ll have a lot more freedom to
experience the entire story of Baldur’s Gate 3 if you make a custom character. With all that in mind...

Which BG3 class (and subclass) is best for you?

Distinctions like caster vs. puncher or melee vs. ranged don’t really work for Baldur’s Gate 3 (or D&D). There’s a lot of overlap in the classes — especially as
you level up.
</item 23>

<item 24>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Their best Ability is Dexterity, so their combat is more martial arts than swordplay. Monks don't get armor and only have a limited set of weapons available,
but they make up for it with Unarmed Strikes that deal extra damage. Monks take a little more finesse to play than Fighters.

Monk subclasses unlock at level 3 and include Way of the Four Elements, Way of the Open Hand, and Way of Shadow. Way of the Four Elements mixes in spells with
your Monk abilities (and uses renamed spells from the caster classes). Way of the Open Hand expands your attacks to include things like knocking down enemies or
pushing them with extra melee attacks. And Way of Shadow makes your Monk sneaky and stealthy (think of it as a ninja subclass).

Way of the Open Palm is the best subclass for a Monk. Consult our Monk build guide for more details on why, and to see which skills and feats to pick.
</item 24>

<item 25>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Best healer class: Cleric

Clerics are the other side of the Paladin’s mix of magic and melee. They’re not quite as frontline in combat with only Light and Medium armors, but they get
more (and more powerful) spells. Clerics also get an extra boost with Channel Divinity that gets them an extra (Divine) magic ability like Turning Undead to
drive away zombies or Guided Strike that gives a +10 to Attack Rolls.

When you make a Cleric, you choose a subclass (Domain) from a list of seven: Life, Light, Trickery, Knowledge, Nature, Tempest, and War. Each Domain has
different spells, actions, and even proficiencies associated with hit.
</item 25>

<item 26>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Minor Illusion Create an illusion that compels nearby creatures to investigate. Poison Spray Project a puff of noxious gas. True Strike Gain Advantage on your
next Attack Roll.

Spells

The Warlock starts with two spells from the Warlock spell list, two of which are exclusive to your subclass.

Baldur’s Gate 3 Warlock Spells Spells Description Spells Description Armor of Agathys Gain 5 temporary hitpoints and deal 5 Cold damage to any creature that
hits you with a melee attack. Arms of Hadar Prevent targets from using reactions. Burning Hands Each flammable target is hit with 3-18 Fire damage. (The Fiend
exclusive) Charm Person Charm a humanoid to prevent it from attacking you. You gain Advantage on Charisma Checks in dialogue. Command Command a creature to
flee, move closer, freeze, drop to the ground or drop its weapon. (The Fiend exclusive) Dissonant Whispers Frighten a creature: it will be easier to hit and
cannot move.
</item 26>

<item 27>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Warlock class features

Hit Points

Hit Dice — 1d8 per warlock level

1d8 per warlock level Hit Points at 1st Level — 8 + your Constitution modifier

— 8 + your Constitution modifier Hit Points at higher levels — 5 + your Constitution modifier per warlock level after 1st

Cantrips

The Warlock starts with two cantrips from the Warlock cantrip list.

Baldur’s Gate 3 Warlock Cantrips Cantrip Description Cantrip Description Blade Ward Take only half the damage from Bludgeoning, Piercing, and Slashing attacks
Bone Chill Prevent the target from healing until your next turn. An undead target receives Disadvantage on Attack Rolls. Eldritch Blast Conjure 1 beam(s) of
crackling energy. Friends Gain Advantage on Charisma Checks against a non-hostile creature. Mage Hand Create a spectral hand that can manipulate and interact
with objects. Minor Illusion Create an illusion that compels nearby creatures to investigate.
</item 27>

<item 28>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
They only get Light Armor, so they have to focus on staying unseen and quick. Their weapons tend to be simple and focus on ones that have the Finesse type —
meaning you can add your Dexterity modifier instead of your Strength. All Rogues get to use Dash (double movement), Disengage (don’t provoke Opportunity
Attacks), and Hide as Bonus Actions. They also get Sneak Attack combat actions that let them deal extra damage whenever they have Advantage.

Level 3 Rogues choose from the Thief, Arcane Trickster, and Assassin subclasses. Thief Rogues get an extra bonus action, Arcane Tricksters mix in magic, and
Assassins get advantages over surprised or unprepared enemies.

Thief is the best subclass for a Rogue. Consult our Rogue build guide for more details on why, and to see which skills and feats to pick.

Best class for melee and magic: Paladin

Paladins start out mixing in (Divine) magic with their melee abilities. They’re the most magic-focused of the melee classes.
</item 28>

<item 29>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
For Warlocks, it’s best to focus on weapon and armor proficiencies:

Drow ’s Drow Weapon Training adds weapons like the rapiers and hand crossbows

’s Drow Weapon Training adds weapons like the rapiers and hand crossbows Humans ’ and Half-Elves ’ Civil Militia add spears, pikes, halberds, and glaives along
with shields

’ and ’ Civil Militia add spears, pikes, halberds, and glaives along with shields Githyanki Martial Prodigy means you get medium armor along with longswords and
greatswords

Martial Prodigy means you get medium armor along with longswords and greatswords Dwarven Combat Training gets you battleaxes, handaxes, light hammers, and
warhammers. The Shield Dwarf subrace also gets you Medium Armor

When you’re creating and leveling up your Warlock, Charisma (for spellcasting) is your most important stat.
</item 29>

<item 30>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
If you’re bound to The Archfey, you’ll gain an action named “Fey Presence,” which will charm or frighten nearby foes.

The Archfey also has two exclusive spells — Faerie Fire and Sleep.

Best Warlock subclass and build in BG3

Baldur’s Gate 3 races don’t really have any direct impact on your class, but certain races and subraces come with features that play well with certain classes
like extra movement or proficiency with certain weapons.
</item 30>
=== END ITEM id=d450d8f80acc1aa0 ===
