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

=== ITEM id=59aff4b6fe831ec8 ===
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
</item 1>

<item 2>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
So many national team fans sided with Reyna during the controversy that developed as the team competed at last winter's World Cup. They got what they argued for
in this game, but not what they wished.
</item 2>

<item 3>
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
</item 3>

<item 4>
How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
Tyler Mitchell Albany, Georgia, 2021 (detail). Credit: Courtesy of the artist and Jack Shainman Gallery, New York As a 20-year-old film student at New York
University, Mitchell became enamoured with still photography after he signed up to do a documentary photography course in Havana, Cuba. Frustrated by the amount
of time and money required to make films, he was drawn to the accessibility of still photography.

Advertisement

“I was impatient when I got started and just wanted to make visual things now, and I didn’t want to wait for permission, or didn’t want to wait on equipment, or
didn’t want to lug large amounts of equipment up flights of stairs. I just wanted to make something that I felt compelled by, immediately,” he says. His career
has only blossomed in the years since he took that trip, and while some artists might feel trepidation at the attainment of so much success from such an early
age, Mitchell is clear-eyed. “The idea of the road ahead only excites me,” he says.
</item 4>

<item 5>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
“And, to be honest, maybe 18, after I didn’t get the last list spot at Collingwood, but I’m glad the stars aligned the way they did.” In a twist of fate,
defender Charlie Dean re-fractured his right foot at training, opening another list spot for Collingwood to snatch Markov back from Carlton – and the rest is
history. “Having trained at Collingwood for two months, I felt like I was almost doing the dirty on ‘Fly’ [Craig McRae], going behind his back – I was almost
like a cheater,” he said. “He was very respectful and understanding, and it probably allowed me to open myself up and be vulnerable to that football club. I’m
super grateful for them showing interest in me, in the first place.

“I know I’ve signed with [Collingwood] and on the dotted lines, I’ve agreed to hate Carlton, but there’s a bit of me that is really grateful and appreciative to
them.”
</item 5>

<item 6>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Continued that form into this season and was outstanding for Western Australia at the National Championships winning his team’s MVP award and All-Australian
honours after averaging 23.5 disposals and 6.5 marks playing as a defender and midfielder. Played a handful of senior games for Claremont and ended the season
being one of his team’s best in a premiership win at under-18 level. His agility, speed for his size and precise kicking are all features of his game.
</item 6>

<item 7>
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
</item 7>

<item 8>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Two years later he joined Southampton in the same role, and that’s where he really began to show off, landing the Saints a number of players who would not only
go on to become Champions League regulars, but make the club a fortune and ensure a Premier League best finish.

Showcasing his skills in casting his scouting net far and wide, Sadio Mane, Dusan Tadic, Graziano Pelle, Dejan Lovren and Victor Wanyama all went on to secure
Southampton a bright future, and Mitchell a promotion to Tottenham in 2014.

6 Mane was one of Mitchell's early successes

6 But there's been many more Credit: Getty

The north London side had just unsuccessfully splurged their world record £85million from Real Madrid for Gareth Bale, and it looked like Mitchell couldn’t have
picked a worse time.
</item 8>

<item 9>
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
</item 9>

<item 10>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Didn’t test at the national Draft Combine due to injury.

Weapon: Neat kicking and ball-winning ability

AFL player comparison: Jack Macrae (Western Bulldogs)

Riley Hardeman

State: Western Australia

State League Club: Swan Districts

Community Club: Caversham

Date of Birth: 27/02/05

Height: 185cm

Left-footed medium defender who is strong overhead, has line-breaking speed and is a penetrating kick. These attributes were on show as captain of Western
Australia at the National Championships where he earned All Australian selection after averaging 18.5 disposals and 5.8 marks and displayed composure under
pressure. Made his senior debut for Swan Districts this year and impressed when representing Australia as part of the AFL Academy.
</item 10>

<item 11>
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
</item 11>

<item 12>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
???https://t.co/nf4CwUbK9A — AFL (@AFL) November 11, 2023

Archer Reid

State: Victoria Country

State League Club: Gippsland Power

Community Club: Inverloch-Kongwak

Date of Birth: 19/09/05

Height: 203cm

A mobile tall forward who is an exceptional overhead mark, good below his knees and has a natural goal sense. Continued to develop physically and improved his
forward craft as the season progressed. Averaged 13.5 disposals, five marks and kicked 16 goals in 11 matches in the Coates Talent League for the Gippsland
Power to finish runner-up in the club’s best and fairest award.
</item 12>

<item 13>
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
</item 13>

<item 14>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
His time of 5:56 seconds for the 2km time trial at the national Draft Combine was sensational for any player much more so for one of his height. Won the Hunter
Harrison Medal for his performances with the Gold Coast Suns Academy. The medal is presented to the best performed player from the non-Victorian Academy teams
who compete in a selection of matches in the Coates Talent League.

Weapon: Running ability and ball use

AFL player comparison: Mark Blicavs (Geelong Cats)

You're going to want to Read more about Ethan! ????



Hear more from Gold Coast Academy ruckman Ethan Read ahead of the #AFLDraft ???
</item 14>

<item 15>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Loading “We just want to win. We come into the club every day and get better, and we just set ourselves up, so when we come to game day, we know we’ve done
everything, to get the result we want.” Maynard had a tumultuous start to September after his infamous collision with Melbourne’s Angus Brayshaw left the Demon
concussed and him defending himself at the tribunal. With a three-match ban hanging over his head, Maynard feared his dream to achieve premiership glory could
be lost before he beat the charge.

Collingwood’s triumph helped banish the pain of their grand final defeat to West Coast in 2018, when Maynard was unfortunate not to receive a free kick for a
block against Willie Rioli in the same passage that led to Dom Sheed’s match-winning goal. “Everyone knows that was a bizarre and crazy sort of week, mentally
[after the Brayshaw incident],” Maynard said. “I wasn’t going well because of the what-ifs, if I missed this.
</item 15>

<item 16>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Composure and step through congestion

AFL player comparison: Adam Cerra (Carlton)

George Stevens

State: Victoria Country

State League Club: GWV Rebels

Community Club: South Warrnambool

Date of Birth: 14/04/05

Height: 189cm

Advertisement

A strong-bodied defender who also plays as an inside midfielder to great effect. He reads the cues to intercept well when playing across half back, is a good
decision-maker and has a penetrating kick. Had an outstanding year which included captaining the Australia under-18 team during the AFL Academy’s matches
against SANFL and VFL sides. Won All Australian honours at the National Championships when representing Victoria Country and was named captain of the Coates
Talent League Team of the Year after averaging 28.8 disposals and five tackles across 13 matches for the GWV Rebels, also winning the club’s best and fairest
award.
</item 16>

<item 17>
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
</item 17>

<item 18>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was an outstanding player for Victoria Country at the National Championships to earn All Australian honours, which included a match-winning performance against
Victoria Metro in the final game where he amassed 22 disposals, 10 marks and four goals. Was also dominant for the Gippsland Power to win their best and fairest
and earn selection in the Coates Talent League Team of the Year. Finished the year with 42 goals across 20 matches in all competitions. His natural leap was
highlighted at the national Draft Combine with his running vertical jump of 97cm ranked second overall. Hails from a talented football family, with his brother
Xavier now at Essendon and sister Yasmin playing for Port Adelaide.
</item 18>

<item 19>
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
</item 19>

<item 20>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
A mobile and competitive player still honing his ruck craft, he is a member of Fremantle’s NGA program with the Dockers able to match a rival club’s bid should
it come after selection 40 in the Draft. Tested well at the national Draft Combine with his running vertical jump of 94cm ranked fifth overall, while his 3.07
seconds on the 20m sprint was impressive for a player of his size.

Weapon: Height and overhead marking

AFL player comparison: Tim English (Western Bulldogs)

Joel Freijah

Pronunciation: Free-jah

State: Victoria Country

State League Club: GWV Rebels

Community Club: Horsham Saints

Date of Birth: 14/11/05

Height: 190cm

A wingman who has real composure, is clean below his knees and distributes the ball very well.
</item 20>

<item 21>
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
</item 21>

<item 22>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
We need to maximize the development of the players. At the end of the day, that’s the key to success and alignment of your resources, for a small country.

“Because Australia, if you look at the history of rugby, when we were amateur, we were probably ahead of the game because we had the influence of rugby league.
We had the influence of AFL and our players were more professional than the rest of the world.”

He added: “I know what’s going on and we need to get a better system of developing players. We need to develop better players and we need to develop a larger
batch of better players and that only comes through changing the whole system.”

While all five Australian Super Rugby sides fundamentally are on board with regards to moving to a centralised high-performance program, The Roar added that
some states, including the ACT Brumbies as well as the QRU, don’t want to give up their commercial assets.
</item 22>

<item 23>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Markov is “a big advocate” for all players being recognised.

“The best thing we can do is wrap our arms around them, and just let him feel what we sort of feel,” he said. “I’ve been on the other side of the fence, and it
is hard because you’re seeing all your friends succeed and sort of living the dream that you aspire to have, but I know it makes people hungrier. I know the
boys who missed out will be the ones who are training the house down, and pushing the boys who have seen the mountain top. They’re going to motivate the boys to
try and see it again.” ‘Incredibly sore’: Howe played out decider with rib injury Peter Ryan

This wasn’t the type of “car crash” injury Jeremy Howe has suffered before, but it was brutal nonetheless. The high-flying Magpie has revealed he played out the
grand final in pain after he was floored by a bone-crunching knock to the ribs from Lions forward Charlie Cameron during the last quarter.
</item 23>

<item 24>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Aerial skill and goal sense

AFL player comparison: Isaac Heeney (Sydney Swans)

Charlie Edwards

State: Victoria Country

State League Club: Sandringham Dragons

Community Club: Sandhurst

Date of Birth: 18/05/05

Height: 191cm

Medium defender and midfielder who emerged as the year progressed and is now considered a possible Draft bolter given his rapid progression for the Sandringham
Dragons in their road to the Coates Talent League premiership. Showed real class with his athleticism and ball use once moved into the midfield with his
29-disposal performance against the Dandenong Stingrays in Round 13 an eye-catcher. Is also a talented cricketer and has spent only one year in the AFL Talent
Pathway system after playing underaged football at Sandhurst before moving to Melbourne Grammar.
</item 24>

<item 25>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Has a sharp competitive edge which has seen him already play 10 senior matches for Subiaco in the WAFL. Tested impressively at the national Draft Combine to
record the second quickest time on the agility test (8.19) and a top ten finish in the 2km time trial (6:17).

Weapon: Power and clean hands

AFL player comparison: Cameron Rayner (Brisbane Lions)

Jed Walter

State: Queensland

State League Club: Gold Coast Suns Academy

Community Club: Palm Beach Currumbin

Date of Birth: 08/06/05

Height: 194cm

A strong and powerful tall forward who has taken all before him across the past two years for the Gold Coast Suns Academy and Allies, winning All Australian
honours at under 18 level in both seasons. Has a real physical presence up forward with his appetite for the contest, along with his second efforts and tackling
pressure.
</item 25>

<item 26>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Height and speed

AFL player comparison: Max Gawn (Melbourne)

"He's got the most upside of any player in our state."



You're going to want to learn more about South Australian Taylor Goad, a 206cm ruckman that is seriously quick.https://t.co/KztP0FJ4wL — AFL (@AFL) November 6,
2023

Advertisement

Phoenix Gothard

State: NSW/ACT

State League Club: Murray Bushrangers

Community Club: Albury

Date of Birth 07/09/05

Height: 178cm

A small forward who came to prominence with his creative forward play for the Allies in their unbeaten National Championship-winning run, where he was a
livewire at ground level and averaged 10.3 disposals, three tackles and a goal a game.
</item 26>

<item 27>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
“I’m super grateful that I’ve got a little chant like that, and hopefully, it’s a little legacy I can continue, and I’m just happy the Magpie army are looking
after me. I feel like they’re always on my side, no matter what mistake I make ... if anything, they motivate me to continue to take the game on and just be my
true self.” Markov felt like he was cheating on Collingwood when he began training with the Blues, after finding out the Pies overlooked him for what was then
their final list spot.

That followed Gold Coast delisting the 27-year-old, who also played 23 games for Richmond. There was interest in him at the time, but he feared his career might
be over. “I went through the national draft and no one looked into me, so there were technically 17 clubs that didn’t really want me,” he said.
</item 27>

<item 28>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
Against Sri Lanka, they were once again exposed.

“Too many of our guys just want to play crash, bang wallop all the time irrespective of the type of pitch or opposition. Again not smart.”

The brunt of Boycott’s criticism of England’s team composition was directed at the continued struggles of Jonny Bairstow, who has averaged just 25.2 this
tournament, and all-rounder Chris Woakes, whose nightmare campaign continued against Sri Lanka with figures of 0/30 from six overs and a fourth-ball duck.

“Bairstow has struggled since his horrific injury [breaking his leg 12 months ago],” Boycott wrote.

“I have said before I believe Jonny came back too soon and was short of fitness and match practice. In the recent five Ashes Tests his keeping at Lord’s was
poor with missed chances and he made only three significant batting scores out of nine innings.
</item 28>

<item 29>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Even if one can argue he has the skill set to succeed at the No. 6, he did not appear comfortable transitioning on short notice.

MORE: USA vs. Germany player ratings

“Tyler’s a great player. It’s not nice to talk about guys that aren't here, though,” Berhalter told reporters afterward. “I’d rather talk about the guys that
are on the field.”

He didn’t, though, relative to how this group filled Adams’ huge vacancy. He was pleased to talk about the pressure the USMNT applied in the first half,
including a ball played over the top of Germany’s defense to Pulisic early in the game that was stopped by an offside call both player and coach disputed, and
Pulisic’s brilliant goal in the 27th minute that involved him driving to his right and blasting a curving shot while four defenders converged and ace goalkeeper
Marc-Andre Ter Stegen watched it sail into the upper righthand corner.
</item 29>

<item 30>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Announced himself as an AFL prospect when he kicked six goals from 26 disposals in the GWV Rebels’ loss to the Dandenong Stingrays in Round 4. He holds his
width when playing on the wing which provides his team with an outlet option with its ball movement. Had a consistent year at club level averaging 19.6
disposals in 14 matches for the Rebels and finished third in the best and fairest while playing in all four matches for Victoria Country. A strong all-round
athlete, he ran a time of 2.98 seconds in the 20m sprint during pre-season testing and then completed the 2km time trial in 6:13 at the national Draft Combine.
</item 30>

<item 31>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
"Last week 45,000 fans (versus Portugal), last night (60,000 in Lyon). Rugby's such a small aspect in terms of Australians but our fans have turned up here,
we've seen them and spoken to them and their unconditional love for us, the fact that they came 20,000km to get here and support us, has been awesome.

"My message is just to understand that we're working extremely hard, there's no stone unturned, especially the youth in this team, to leave the jersey in a
better place and just to stick with us.

"I know the results haven't been coming our way the last couple of years but when it does turn, that's what we're building towards.﻿"
</item 31>

<item 32>
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
</item 32>

<item 33>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Advertisement

“What’s changed back there that I could think that, what I can offer, can change what they’re doing still?

“I’ll always love rugby. I’ll always love Australian rugby. Could I see myself working in Australia again? I doubt it.”

But RA have been making moves to try and twist Nucifora’s arm.

While Nucifora was never in the running for the director of high-performance, it’s understood they are trying to get him on board as a consultant when he wraps
up with Ireland.
</item 33>

<item 34>
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
</item 34>

<item 35>
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
</item 35>

<item 36>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
A very clean ball handler with a dazzling burst of speed, he was a consistent performer across all four matches for the Allies, averaging 16.5 disposals at 80
percent efficiency. His speed and power really shone through at the national Draft Combine where he was ranked second in the 20m sprint (2.92 seconds) and
standing vertical jump (78cm) as well as being third in the running vertical jump (95cm) which confirmed his elite athletic profile.

Weapon: Speed and ball use

AFL player comparison: Dan Houston (Port Adelaide)

Will Green

State: Victoria Metro

State League Club: Northern Knights

Community Club: Northcote

Date of Birth: 08/09/05

Height: 204cm

Developing tall who impressed throughout the year with his ruck craft, marking and mobility when playing for the Northern Knights, Victoria Metro and the
Australia under-18 team.
</item 36>

<item 37>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Showcased his athletic prowess at the national Draft Combine where he produced the quickest time on the agility test (8.16 seconds) and a top ten finish on the
20m sprint (2.92 seconds). Is a member of the West Coast Eagles’ Next Generation Academy meaning they can match any bid after selection 40 in this year’s Draft.

Advertisement

Weapon: Speed and innate goal sense

AFL player comparison: Charlie Cameron (Brisbane Lions)

Jordan Croft

State: Victoria Metro

State League Club: Calder Cannons

Community Club: Maribyrnong Park

Date of Birth: 12/05/05

Height: 200cm

A tall forward who is lightly built but extremely athletic. He uses his mobility and height to create opportunities as a marking forward, with his speed and
precise kicking key features of his game. He only fully focused on a potential AFL career last year after previously playing volleyball.
</item 37>

<item 38>
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
</item 38>

<item 39>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
An excellent decision-maker and ball user by hand or foot, he averaged 19.5 disposals, 12.5 contested possessions and 6.5 clearances to finish third in the
Larke Medal voting and took out Victoria Country’s MVP. Further added to his very impressive record by earning selection in the All Australian team for the
second year in a row. Handled the step up to playing senior football with three impressive games in the VFL and was the best player for the Australia under-18
team in its match against Port Adelaide’s SANFL team.
</item 39>

<item 40>
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
</item 40>

<item 41>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Tested very well at the state Draft Combine in Victoria with his 6:12 on the 2km time trial and 3.04 seconds for the 20m sprint rounding off a productive
season. Is eligible to be selected under the father-son rule as the son of former Melbourne utility Nathan D Brown, who played 146 games for the Demons.

Weapon: Footy IQ and ball-winning ability

AFL player comparison: Errol Gulden (Sydney Swans)

Cal Twomey believes father-son prospect Kynan Brown has shown enough to be on an AFL list in 2024, but is more likely to be a later selection in the Draft
</item 41>

<item 42>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Progressed from dominating under-18 games at club level to playing seven matches at reserves level averaging 18.7 disposals. Featured among South Australia’s
best in the National Championship matches against the Allies and Victoria Metro and averaged 18 disposals, 3.8 clearances and 8.5 tackles across the tournament.
Showcased his power and speed at the national Draft Combine with his running vertical jump of 95cm ranking third and 2.93 seconds for the 20m spring ranking
fifth.

Weapon: Power and left-foot kick

Player Comparison: Jack Crisp (Collingwood)

Will McCabe

State: South Australia

State League Club: Central District

Community Club: Tanunda

Date of Birth: 29/09/05

Height: 197cm

A tall defender who has also gone forward at times this year and has continued to develop nicely. A competitive type who has a good natural leap and is very
strong overhead.
</item 42>

<item 43>
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
</item 43>

<item 44>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
CLICK HERE for a seven-day free trial to watch the AFL on KAYO

KEVIN SHEEHAN’S TOP 40 PROSPECTS: VIA THE AFL

Kynan Brown

State: Victoria Metro

State League Club: Oakleigh Chargers

Community Club: Blackburn

Date of Birth: 13/01/05

Height: 181cm

Left-footed inside midfielder who has quick hands, sharp vision and creates play through the middle of the ground with his creativity. Had a remarkably
consistent year which included starring for Victoria Metro in the National Championships averaging 22.8 disposals at 80 percent efficiency as well as 4.5
clearances and 4.8 tackles to win their best and fairest award. Had similar numbers with the Oakleigh Chargers in the Coates Talent League including kicking
five goals and having 25 disposals in a best on ground performance in the Wildcard Round win against the Northern Knights.
</item 44>

<item 45>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
“Some of those cricketers on that park today would go in your best ever England white ball XI. It’s just been a bridge too far for most of them.

“What I don’t like is giving players a cop out… when we win the 50-over World Cup and the 20-over World Cup, aren’t they great? We’re brilliant. And when the
wheels come of it’s the structure. It’s the structure of English cricket, we’re a disgrace.

“We play 20-over cricket, we play 100-ball cricket, we don’t play enough 50-over cricket – how much 50-over cricket domestically has Virat Kohli played? Or
Heinrich Klaasen or anyone out here?

“That’s what’s made this great side over the last six years, travelling round the world playing T20 franchises. It’s such a lame excuse.”
</item 45>

<item 46>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Clean hands and ball use

AFL player comparison: Daniel Rioli (Richmond)

Jake Rogers

State: Queensland

State League Club: Gold Coast Suns Academy

Community Club: Broadbeach

Date of Birth: 02/03/05

Height: 170cm

A tenacious livewire through the midfield, the Gold Coast Suns Academy member is a one-touch player with great vision and creativity. Showed all those traits at
the National Championship by starring for the Allies where he consistently set up his team through the middle of the ground. Quick and with elite agility, he
was always in the contest. Averaged 22.5 disposals at 77 percent efficiency, 5.5 clearances and 5.2 tackles to be a driving force for the title-winning team.
</item 46>

<item 47>
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
</item 47>

<item 48>
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
</item 48>

<item 49>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Adams and McStay – who arrived from the Lions in the off-season as an unrestricted free agent – leaned on each other to cope with the difficult moment, and
needed some privacy to deal with their personal misfortune before they could share in the team’s raucous celebrations. McStay may need as many as eight weeks to
recover from the high-grade medial collateral ligament strain he suffered in the club’s preliminary final win. Collingwood’s injured recruit Dan McStay at
training before the grand final. Credit: Eddie Jim “We shed a few tears and a bit of emotion, then we brushed it off and said, ‘Let’s get on with it and go
celebrate with the boys’. It was a great night,” Adams said. John Noble was another sad story, playing every game in the home-and-away season, only to be
dropped for Collingwood’s qualifying final clash with Melbourne and never forcing his way back in.
</item 49>

<item 50>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
With moments left in extra time against the South Americans, Arnold dragged Mat Ryan and replaced him with Andrew Redmayne – who saved the decisive spot kick
(and also threw his opponents water bottle with cheat sheet attached away from the goal).

Advertisement

Ryan might well have gotten the Socceroos to Qatar, but if Australia had not advanced there’s little chance Arnold would still be their coach today.

And there’s virtually no chance he’ll have to endure a similar level of stress through to the next finals – with FIFA expanding the tournament from 36 to 48
teams meaning eight direct Asian confederation qualifiers will proceed directly instead of four.

Arnold, in his second stint as national team manager, will prowl the technical area for a 59th time on Thursday, taking him clear of Frank Farina.
</item 50>

<item 51>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Nathan Murphy, who took no further part in the grand final after being concussed in a clash with Brisbane’s Linc McCarthy, spared a thought for all three
players amid the post-match chaos. “It was pretty emotional seeing them. You go from one high like myself to a low like theirs,” Murphy said. “You’ve got to put
yourself in their shoes, and see that, but they’re so good. They’re turning it into a positive, and they’re just happy for everyone else’s success, so it’s
awesome.” The trio’s grand final heartbreak sparked renewed debate about whether every player on the list should receive a premiership medallion, or at least
those who played a certain number of games. First-year Magpie Oleg Markov has experienced both sides of the ledger. He played in Saturday’s premiership, but
missed out on all three Richmond flags (2017, 2019 and 2020) despite playing senior games in each of those years.
</item 51>

<item 52>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.
</item 52>

<item 53>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Marking on the lead and goalkicking accuracy

AFL player comparison: Bayley Fritsch (Melbourne)

Ollie Murphy

State: Victoria Metro

State League Club: Sandringham Dragons

Community Club: East Brighton Vampires

Date of Birth: 16/09/05

Height: 200cm

A lightly built tall defender who sets up the play with his neat left-foot kicking. Although new to playing in defence this season he adapted remarkably well.
Shows real composure with the ball in hand and is strong overhead. Is also a capable decision maker on when to mark or spoil in one-on-one contests. Was
outstanding for Victoria Metro in the National Championships and won his team’s MVP award along with All Australian honours after averaging 11.2 disposals and
4.5 marks per match.
</item 53>

<item 54>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Shared the role of Victoria Country co-captain with Harley Reid and performed consistently. Had an exciting year with the Dandenong Stingrays averaging 19
disposals and 4.8 tackles to earn selection in the Coates Talent League Team of the Year. Recorded a top 10 finish in the 20m sprint at the national Draft
Combine with a slick 2.99 seconds. A dual sports athlete who was a member of Australia’s under-19 cricket program.

Weapon: Speed and endurance

AFL player comparison: Jordan Dawson (Adelaide Crows)

Zane Duursma

State: Victoria Country

State League Club: Gippsland Power

Community Club: Foster

Date of Birth: 28/08/05

Height: 189cm

A mercurial forward or midfielder who reads the play exceptionally well, is strong overhead and deadly accurate around the goals.
</item 54>

<item 55>
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
</item 55>

<item 56>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
???



The 200cm forward is an elite runner, sharp kick and strong mark.#AFLDraft pic.twitter.com/f9twMnNtfO — AFL (@AFL) October 12, 2023

Daniel Curtin

State: Western Australia

State League Club: Claremont

Community Club: West Coast

Date of Birth: 08/03/05

Height: 197cm

A left-footed defender who experienced a frustrating run with injury during the 2022 season. Managed to get back to play in the AFL Futures game at the MCG on
AFL Grand Final day and excelled for Team Houli with 22 disposals and nine marks to claim best on the ground honours.
</item 56>

<item 57>
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
</item 57>

<item 58>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
He capped off a great year playing in the Sandringham Dragons premiership team in the Coates Talent League.

Weapon: Height and intercept marking

AFL player comparison: Jake Lever (Melbourne)

Connor O’Sullivan

State: NSW/ACT

State League Club: Murray Bushrangers

Community Club: Albury

Date of Birth: 19/05/05

Height: 198cm

Advertisement

Athletic tall defender who was outstanding for the title-winning Allies at the National Championships winning their MVP award and All Australian honours. He
averaged 17.8 disposals at 86 percent efficiency as well as averaging 8.2 marks and providing great rebound to lead the Allies defence. He has an exciting
natural leap, good speed and elite endurance highlighted with a top 10 placing in the 2km time trial at the national AFL Draft Combine with a time of 6:11.
</item 58>

<item 59>
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
</item 59>

<item 60>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Kicked 39 goals across all matches in the talent pathway this year, which included games for Victoria Metro and the Australia under-18 team. Tested extremely
well at the national Draft Combine with results of 3.02 seconds on the 20m sprint along with 6:35 for the 2km time trial indications of his athletic prowess. Is
the nephew of two-time Richmond premiership star Josh Caddy.

Weapon: Contested marking and goalkicking

AFL player comparison: Taylor Walker (Adelaide Crows)

Caiden Cleary

State: NSW-ACT

State League Club: Sydney Swans Academy

Community Club: Sydney University

Date of Birth: 05/03/05

Height: 182cm

An in-and-under player who extracts the ball well from stoppages.
</item 60>

<item 61>
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
</item 61>

<item 62>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
His love for the Socceroos and football in Australia is both infectious and inspiring. That energy and passion is a huge element to the way he’s able to lead
and inspire our nation’s best male footballers in some of our sport’s most challenging arenas.

“On behalf of Football Australia, I’d like to congratulate Arnie on everything he’s achieved in reaching this milestone and we look forward to him continuing to
build on what is already an exceptional legacy.”

In a remarkable few weeks last year Arnold went from on the nose to a man revived. Since then he’s been linked to club jobs in Europe but opted to stick around
with the national team.

While national team coaches make important calls every game, Arnold’s record-breaking achievement is conceivably down to one big ballsy call in the playoff
against Peru.
</item 62>

<item 63>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Power and contested ball-winning ability

AFL player comparison: Dustin Martin (Richmond)

Archie Roberts

State: Victoria Metro

State League Club: Sandringham Dragons

Community Club: Hampton Rovers

Date of Birth: 18/11/05

Height: 184cm

A left-footed half-back who is a penetrating kick, smart interceptor and uses his anticipation and clean hands to have an impact. Showed good rebounding speed
and fine decision-making ability to set up play from defence for Victoria Metro during the National Championships, where he averaged 20.8 disposals and 3.8
marks to earn All Australian honours before becoming a back-to-back premiership player for the Sandringham Dragons in the Coates Talent League. His impressive
time of 6:08 for the 2km time trial at the national Draft Combine gave a glimpse of his elite endurance and competitiveness.
</item 63>

<item 64>
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
</item 64>

<item 65>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
He only fully focused on a potential AFL career last year after previously playing volleyball. Starred with the Calder Cannons and Victoria Metro this year
kicking 33 goals in the talent pathway. His 2km time of 6:05 at the national Draft Combine was a strong indicator of his elite endurance but also his
competitiveness for a player of his size. Is eligible to join the Western Bulldogs under the father-son rule as the son of former Bulldogs utility Matthew
Croft, who played 186 games for the club.

Weapon: Aerial ability

AFL player comparison: Max King (St Kilda)

Jordan Croft has nominated as a father-son prospect for the Western Bulldogs ????
</item 65>

<item 66>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was also named captain of the All Australian team after leading the Allies to their first title. Caught the eye with a strong performance for the Australia
under-18 team in its representative match against Port Adelaide’s SANFL side and backed it up in the second match against Carlton’s VFL side where he collected
24 disposals to be one of Australia’s best in victory. Finished the year as a back-to-back Coates Talent League premiership player with the Sandringham Dragons
where he averaged 31 disposals across nine matches.
</item 66>

<item 67>
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
</item 67>

<item 68>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
We felt like he was a father figure for us more than a coach.” Markov’s actual father, Dmitri, famously won the men’s pole vault at the 2001 world
championships, so his role in the Magpies’ triumph continues the family’s sporting success. “I’m very lucky. Dad thinks it’s a bit harder in a team sport
because everyone has to perform,” Markov said. “He said, ‘Hey, AFL is only in Australia, but, technically, you are a world champ,’ so I think I’ve matched him
for now. I’d like to get another one, and hopefully, I can one-up him, but I think he’s got the cake at the moment.” Pies’ premiership ‘just the beginning’:
Maynard

Marc McGowan

Brayden Maynard has warned Collingwood’s breakthrough premiership is just the beginning as they bid to become the AFL’s next dynasty team.
</item 68>

<item 69>
How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
Tyler Mitchell: “I’ve never been to Australia so it’s very exciting to sort of have a global or new conversation on a new continent with my work.″⁣ Credit:
Courtesy of the artist and Jack Shainman Gallery, New York The theme of the Triennial is “Magic, Matter, Memory”, making Mitchell’s pieces a logical inclusion.
“A lot of my work meditates on being between a heightened reality, or potentially even a magical reality and an actual reality,” he says.

Advertisement

“I’m creating what I call lightly staged moments. And so that means I’m setting up scenarios where these families are enjoying real moments of leisure in
outdoor spaces, usually open pastures and fields, and those moments are meant to stand in for a meditation on what it means to enjoy leisurely outdoor nature as
a black person.” Tyler Mitchell, Vastness, 2021.
</item 69>

<item 70>
‘Right people in right roles’: QRU breaks silence on RA’s reform hopes, backs review into Wallabies’ RWC flop (The Roar | Sports Writers Blog,
2023-10-10T23:06:29+00:00)
RA is also in the early stages of looking to renew its next broadcast deal, with the governing body hoping to at least double its current deal with Nine
Entertainment/Stan Sport, which is seeing them paid $29m per year until the end of 2025.
</item 70>

<item 71>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Played all four games for Victoria Metro at the National Championships and was his team’s best player in the loss to the Allies, while also shining with 22
disposals in the win over Western Australia. Across his 16 matches with the Eastern Ranges in the Coates Talent League he averaged 19.4 disposals to be one of
their most consistent players during their run to the grand final. Recorded a slick time of 2.92 seconds on the 20m sprint during the national Draft Combine
highlighting his dazzling acceleration and backed it up by recording 6:28 for the 2km time trial which demonstrated a combination of elite speed and endurance
which are essential to play on the wing.

Weapon: Elite speed and ball use

AFL player comparison: Nick Martin (Essendon)
</item 71>

<item 72>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
We’re not at that level. And we can’t pretend to be at that level. But can we be at that level by 2027? Yes, we can,” said Jones.

“We’re definitely off the pace but the opportunity to get better is there. I love the game. I love working with young players. I loved seeing some of the young
players maturing during the World Cup and some of them struggle. And that’s all part of it. But I love the game and I love coaching the Wallabies.”

Jones was named Wallabies coach at the start of the year as replacement for Dave Rennie. He came in with grand statements that his team could win the Rugby
Championship, Bledisloe Cup and make a smash and grab on the World Cup.
</item 72>

<item 73>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Has already played nine matches at senior level with Peel Thunder averaging 18.9 disposals. Starred when he represented Western Australia at the National
Championships averaging 27.2 disposals, 11 contested possessions, six clearances and five tackles to win All Australian honours. Is the son of former West Coast
Eagles and Geelong Cats forward Derek Hall but is not eligible as a father-son to either club given Derek played only two games for the Eagles and 74 matches
for the Cats. Didn’t test at the national Draft Combine due to injury.
</item 73>

<item 74>
How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
“The idea of the road ahead only excites me,” he says. “The idea that there will be very quiet, sleepy periods, and frenetic periods, or hyperactive periods …
It’s always refreshing as a young person to see how the career of a much older, more senior artist has really gone. “That senior artist that we know and love
today had decades where they were out of style, out of fashion, people thought their work was done, their best work was behind them, and yet, often had
resurgences. Loading “I think when we’re thinking about crafting a career … we’re really talking about allowing space for bad and good, allowing space for
artists to make and work through complicated and sometimes not-so-favourable ideas. And I try to give myself that same space. It’s only about giving yourself
grace and space and thinking one day at a time.” Asked if there’s a dream project, or a dream collaborator for him, Mitchell says there are too many.
</item 74>

<item 75>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
His ferocious attack on the ball and his opponent helped set the tone for the Allies to dominate the midfield battle throughout the National Championships and
he was rewarded with All Australian honours. Averaged 29.2 disposals, 11.2 contested possessions and a team-high 6.8 tackles across four matches for the Allies
and showed excellent repeat speed and courage as well as clean hands playing both on the inside and outside. His 6:06 effort in the 2km time trial at the
national Draft Combine showcased his elite endurance. He has been part of the Sydney Swans’ Academy since the under-10s.
</item 75>

<item 76>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
It was reported during the campaign that Jones had an interview for the Japan head coach role but he denied that again on Tuesday, having previously also denied
the story.

“I haven’t been speaking to anyone, mate,” was all Jones would say on multiple reports of conversations with sources in Japanese rugby.

He said he understood why Australian fans were hurting.

“We feel all the supporters hurt and really appreciate the great support we had in France and back in Australia for the team,” said Jones.

Advertisement

“Sometimes the change you have to have is uncomfortable. It is quite difficult. And sometimes you have to go through this to get into a better state and I
believe that the problems we’ve had particularly at the national team will put us in a better state going forward.”

Jones clearly believes he is the right man to carry Australia on.

“I think I’ve got the foresight to see where we need to go. I’ve got the experience of being in difficult situations before,” Jones said.
</item 76>

<item 77>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
“Morgan’s England team was a finely tuned machine with everyone knowing their role and most of them at the top of their form… all Morgan had to do was pull the
strings of a great unit of cricketers and he did it well, but my Mum could have captained that lot.

“On the other hand Jos has a dysfunctional squad of players. Poorly selected, poorly prepared, not a settled team, many not sure of their roles, patchy form,
confidence wobbly and to cap it all England’s best two and most influential white-ball players have not been available.”

Boycott’s voice was added to by former England Test captains Nasser Hussain and Michael Atherton, who have both decried the sudden decline of the team’s
previously dominant white-ball set-up, having entered the World Cup as reigning ODI and T20I champions.
</item 77>

<item 78>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
NEW PODCAST! Michael Atkinson and Sam Worthington preview a thrilling final round of pool games to decide the quarter-finalists at the Rugby World Cup in
France﻿

He wanted to regain form at club level ﻿with Urayasu D-Rocks but his hand remained up for Test selection.

"I chose to leave Australian shores and understand the ruling around that and what it holds," he said.

"I've got to perform for my club first and if you get the call up, you get the call up. If you don't, I'll take my three months break and enjoy that."

Kerevi is an unabashed Jones fan.

Eddie Jones with fans during a Wallabies training session at Stade Roger Baudras. (Getty)

"He's up at 4am working hard. We catch him in the gym, he's working hard and I swear he doesn't even sleep.
</item 78>

<item 79>
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
</item 79>

<item 80>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
His natural leap was on display at the national Draft Combine where he featured in the top 10 performers in both the standing vertical jump (75cm) and running
vertical jump (91cm).

Weapon: Aerial ability and left-foot kicking

AFL player comparison: Hayden Young (Fremantle)

Angus Hastie

State: Victoria Country

State League Club: Geelong Falcons

Community Club: Geelong West Giants

Date of Birth: 19/09/05

Height: 190cm

An athletic medium defender who sets the play up from defence with his run and ball use. Has shown a preparedness to back himself and take the game on to create
opportunities for his team. His intercepting ability is also a strength which saw him represent Victoria Country in all four matches at the National
Championships. Played 13 matches with the Geelong Falcons in the Coates Talent League this year averaging 20 disposals and 4.6 marks.
</item 80>

<item 81>
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
</item 81>

<item 82>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Intercept ability and long kick

AFL player comparison: Callum Wilkie (St Kilda)

Cooper Simpson

State: Victoria Country

State League Club: Dandenong Stingrays

Community Club: Mt Martha

Date of Birth: 13/02/05

Height: 181cm

An ultra professional midfielder with pace, composure and great finishing skills. Also has a neat step through traffic and runs hard both offensively and
defensively. Has shown the ability to perform when challenged and in the big moments when the game is on the line. Started the season in good touch with the
Dandenong Stingrays in the Coates Talent League but suffered an ankle injury in Victoria Country’s opening game of the National Championships and was unable to
play any further matches at that level. Made an impressive return for the final four matches with the Stingrays including having 28 disposals against the
Tasmania Devils which reminded scouts of his class and AFL potential.
</item 82>

<item 83>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Holds his ground very well both under the ball in stoppages and marking contests and provides great opportunities for his teammates at ground level. Averaged 12
disposals and 24 hitouts when representing Victoria Metro at the National Championships to earn All Australian honours. Both his running vertical jump of 87cm
and 2km time of 6:34 seconds at the national Draft Combine were impressive indicators of his natural leap and endurance.

Weapon: Mobility and marking

AFL player comparison: Brodie Grundy (Sydney Swans)

Clay Hall

State: Western Australia

State League Club: Peel Thunder

Community Club: Harvey Brunswick Leschenault

Date of Birth: 12/05/05

Height: 189cm

Advertisement

Prolific disposal winner who uses the ball effectively on his preferred left foot and excels at stoppages using his clean hands and vision to create for his
team.
</item 83>

<item 84>
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
</item 84>

<item 85>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
His agility, speed for his size and precise kicking are all features of his game.

Weapon: Intercept ability and precise ball use

AFL player comparison: Steven May (Melbourne)

Advertisement

Harry DeMattia

Pronunciation: De-matt-e-ya

State: Victoria Country

State League Club: Dandenong Stingrays

Community Club: Edithvale-Aspendale

Date of Birth: 11/09/05

Height: 185cm

A midfielder/forward with a combination of speed and endurance which enables him to cover the ground and compete in the air and at ground level. A left footer,
he brings real pressure with his ability to lay run down tackles against the flow of play. Shared the role of Victoria Country co-captain with Harley Reid and
performed consistently.
</item 85>

<item 86>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
Arnold – who also represented the Socceroos 56 times as a player – has been involved in national team coaching ranks since 2000, first as an assistant to Farina
and then Guus Hiddink, before a brief period as caretaker coach in 2006.

“It’s certainly a proud moment and an achievement that I’ll reflect on when the time is right,” Arnold said.

“I look at some of the names on that list – Farina, Arok, Thompson, Rasic – guys I played under or assisted in the early part of my coaching career, and it
gives you some perspective on what a journey it’s been so far.

Advertisement

“I think back to that caretaker period and how much I learnt about coaching; it was a challenging experience and one that on reflection I probably wasn’t ready
for. I’m grateful for the opportunity I was given and what it was able to teach me ahead of my pathway into club coaching.
</item 86>

<item 87>
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
</item 87>

<item 88>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Intercept marking

AFL player comparison: Will Day (Hawthorn)

Advertisement

James Leake

State: Tasmania

State League Club: Tasmania Devils

Community Club: Launceston

Date of Birth: 02/10/05

Height: 187cm

Medium defender and forward who won All Australian honours as a defender for the Allies at the National Championships in the middle of the season then played
some eye-catching matches as a forward for the Tasmania Devils in the Coates Talent League during a breakout season. With mercurial ability in the air and
uncanny ability around goals he looks to have unique versatility which was on display during 11 matches for the Devils where averaged 16.1 disposals and kicked
17 goals, which included a five-goal haul against the Dandenong Stingrays in Round 17.
</item 88>

<item 89>
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
</item 89>

<item 90>
NBL Round 4: Creek pours in points as Phoenix swamp Sixers, CJ claims Kings didn’t look after DJ (The Roar | Sports Writers Blog, 2023-10-19T21:03:08+00:00)
Advertisement

But when the Washington Wizards waived him just before pre-season training camp, the 26-year-old decided to return to the NBL – signing with the Sixers on
Tuesday, which didn’t please Sydney’s powerbrokers.

“We originally amended the contract in good faith and completed our roster with the understanding that ‘DJ’ (Vasiljevic) would play internationally this
season,” Kings chief executive Chris Pongrass said.

“We are disappointed with how this has been handled and another NBL team’s role in this process.”

Australian basketball legend and Kings co-owner Andrew Bogut chimed in on X where he posted, in part, that Sydney had granted Vasiljevic a release “after he
told us he wanted to ‘chase his NBA dream’, not his NBL dream with another club”.
</item 90>

<item 91>
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
</item 91>

<item 92>
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
</item 92>

<item 93>
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
</item 93>

<item 94>
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
</item 94>

<item 95>
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
</item 95>

<item 96>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
AFL Talent Ambassador Kevin Sheehan has revealed his top 40 prospects ahead of this month’s AFL Draft.

Sheehan’s picks were announced in a press release by the AFL on Thursday. The 2023 draft will be held on November 20 and 21 with the rookie draft to follow a
day later.

Sheehan has profiled all 129 players who attended the 2023 AFL Draft Combine or 2023 AFL State Draft Combine last month. His full list can be found at the AFL
Draft Hub.

Facebook Twitter Whatsapp Reddit Email Share

The AFL Draft will be shown on Fox Footy and Kayo.
</item 96>

<item 97>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was impossible to contain during this year’s National Championships booting 14 majors in four matches for Victoria Metro to top the goal kicking. He also
averaged 12.2 disposals and was rewarded with All Australian selection. He further expanded his game by spending more in the midfield and across half back for
the Eastern Ranges and averaged 20 disposals while kicking 11 goals in nine matches in the Coates Talent League.

Weapon: Speed, agility and leap

AFL player comparison: Shai Bolton (Richmond)

Advertisement

This is just what Nick Watson does ????
</item 97>

<item 98>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Uses his dazzling speed to break the lines and kicks the ball with pinpoint accuracy on his preferred left foot. Averaged 32.8 disposals and 5.8 clearances
across four matches for the Allies at the National Championships to earn All Australian honours. Was equally dominant for the Tasmania Devils in the Coates
Talent League having more than 30 disposals in seven of his nine matches and also kicked 12 goals. Showed he could match it against more experienced opponents
by having 20 disposals and three clearances in Australia’s win over Carlton’s VFL team.
</item 98>

<item 99>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
Seriously, the amount of time he puts into effort, into his staff, into the players, always has time for us ... and I think fans don't see that," he said.

"At the end of the day, we didn't get the results, you know, we didn't perform the way we wanted to as a team and it's not on Eddie - it's on all of us.

"So I understand what the fans feel and they're just passionate and it's awesome to have that. But no-one realises what goes on inside. No-one realises the
things that we're working on tirelessly, to work hard, but especially him and the staff, f--ing work so hard, bro.

Samu Kerevi during a Wallabies training session at Stade Roger Baudras. (Getty)

"They're away from families, they're sacrificing as well and … I don't read into the comment stuff, but the young boys, I hear a lot that's going on, social
media and stuff.
</item 99>

<item 100>
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
</item 100>
=== END ITEM id=59aff4b6fe831ec8 ===
