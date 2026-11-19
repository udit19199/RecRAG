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

=== ITEM id=b1ff85463894f4dd ===
Gold sentences and facts:
Sentence 0: It could, though, mean some sort of progress, if the Manchester born Ratcliffe follows through with his vision to revolutionise the club.
  Fact 0.0: It could mean some sort of progress.
  Fact 0.1: Ratcliffe was born in Manchester.
  Fact 0.2: Ratcliffe has a vision to revolutionise the club.
  Fact 0.3: The progress would happen if Ratcliffe follows through with his vision.
Sentence 1: Manchester United are out of Europe altogether after a limp display against Bayern at Old Trafford, where Kingsley Coman sealed a 1-0 win for Thomas Tuchel's side.
  Fact 1.0: Manchester United are out of Europe altogether.
  Fact 1.1: Manchester United gave a limp display against Bayern at Old Trafford.
  Fact 1.2: Kingsley Coman sealed a 1-0 win at Old Trafford.
  Fact 1.3: Kingsley Coman's 1-0 win was for Thomas Tuchel's side.

Retrieved items:
<item 1>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
That raises doubts over the future of current sporting director John Murtough and chief executive Richard Arnold, although manager Erik ten Hag is believed to
be on a solid footing despite United’s poor start to the season.

Expect Sir Dave Brailsford, the mastermind behind Team GB’s cycling success at the London 2012 Olympics and director of sport for Ratcliffe’s company INEOS, to
have a prominent role in any reshuffle.

Ratcliffe also has ambitious plans to upgrade Old Trafford and turn it into a 90,000-seat stadium.

He wants the deal ratified in time for the January transfer window, however due to FFP concerns, there is no guarantees United will be able to spend big.

Despite the fact he turns 71 a day before the board meeting, it appears Ratcliffe’s plans centre around long-term stability and success rather than a short-term
turnaround in United’s fortunes.

Graph facts:
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Erik ten Hag
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Erik ten Hag
- Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win -MENTIONS-> Erik ten Hag
- Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign -MENTIONS-> Erik ten Hag
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Erik ten Hag

Entities: Erik ten Hag, Man United, Old Trafford, talkSPORT, Sir Jim Ratcliffe, Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim
Ratcliffe as he seeks board vote, Man United takeover, John Murtough, Richard Arnold, Sir Dave Brailsford, Team GB, London 2012 Olympics, INEOS, FFP concerns,
January transfer window
</item 1>

<item 2>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
The manager’s future

Erik ten Hag’s team are treading water at precisely the wrong moment: as Sir Jim Ratcliffe takes charge of Manchester United’s sporting department before making
sweeping changes in the posts that will directly affect the manager. Richard Arnold has left despite officially staying until the end of the year and into his
role may come Jean-Claude Blanc, the CEO of Ratcliffe’s Ineos Sport, as the man Ten Hag reports to. Ten Hag’s line manager may change too, as Ratcliffe
considers whether a new football director is required, a position held by John Murtough. Ten Hag may have a first clue of his status in Ratcliffe’s new empire
if he is sounded out for his view of a prime factor in his ability to coach and manage the side to success: United’s transfer policy. This leads us to Murtough
…

Will Murtough survive?

The football director has no plans to depart but whether he remains in situ may be a matter for Ratcliffe.

Graph facts:
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Kasper Schmeichel
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Ross Barkley
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Murtough
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Aaron Ramsey
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Leicester

Entities: Jean-Claude Blanc, Ineos Sport, transfer policy, From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray, Manchester United,
Erik ten Hag, The Guardian, Sir Jim Ratcliffe, John Murtough, Richard Arnold
</item 2>

<item 3>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Jim Ratcliffe’s move to Monaco in 2020 may have been bad news for the British taxpayer, but not for Manchester United fans.

The chemical billionaire looks set to buy a 25 per cent stake in the Premier League giants after fellow bidder, Sheikh Jassim, withdrew his offer.

6 Ratcliffe could still have a big say at United Credit: getty

6 His wisest move could be to land his old neighbour Mitchell

That news hasn’t been too warmly welcomed by United supporters, who would still have to put up with the hugely unpopular Glazer family in control of the club.

It could, though, mean some sort of progress, if the Manchester born Ratcliffe follows through with his vision to revolutionise the club.

Should Ratcliffe come in, talkSPORT understands that he would take charge of footballing matters, something the Glazers have been accused of having little to no
interest in.

Graph facts:
- Everton players and fans will be fired up by points penalty, asserts Sean Dyche -MENTIONS-> Chelsea
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Chelsea
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Chelsea
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Chelsea
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Chelsea

Entities: Chelsea, Manchester United, Manchester, Daniel Levy, Jim Ratcliffe, Glazer family, Monaco, footballing matters, Jim Ratcliffe's move to Monaco, Paul
Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United, Paul Mitchell, talkSPORT, Sheikh
Jassim, Tottenham
</item 3>

<item 4>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
This was a move mainly influenced by the fact Raine had managed to broker a deal with Todd Boehly to commit to invest £4.5bn to take control of Chelsea, giving
the Glazers grand ideas they could rake in as much as DOUBLE for a global sporting institution like United.

After several rounds of bidding, it quickly became clear the race for United was a straight shootout between Oldham-born Ratcliffe, who had previously offered
£4bn to buy Chelsea, and Qatari Sheikh Jassim bin Al Thani, who also professed to being a lifelong Red Devils fan and set up the ’92 Foundation’ in tribute to
the famous class of ’92 of David Beckham and company.

Graph facts:
- Everton players and fans will be fired up by points penalty, asserts Sean Dyche -MENTIONS-> Chelsea
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Chelsea
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Chelsea
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Chelsea
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Chelsea

Entities: Chelsea, Man United, talkSPORT, Sir Jim Ratcliffe, Glazers, Raine, Todd Boehly, Sheikh Jassim bin Al Thani, ’92 Foundation, class of ’92, David
Beckham, Bidding race for United, Todd Boehly deal to invest in Chelsea, Oldham, Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim
Ratcliffe as he seeks board vote
</item 4>

<item 5>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
5 Joel and Avram Glazer put the Red Devils up for sale in November 2022 Credit: AFP

5 United supporters are desperate to see the Glazers depart Old Trafford Credit: Twitter

The fact the Glazers will remain in situ has not gone down well with supporters, but this is seen as the first step to Ratcliffe eventually removing the
Americans and taking overall control.

The exact timescale as to when that happens remains vague and how much it will cost Ratcliffe overall could depend on United’s sporting and commercial success
in the coming seasons.

Thursday’s board meeting was pre-arranged and while it is expected that Ratcliffe’s proposal will discussed in detail, sources close to United have indicated to
talkSPORT that it is far from guaranteed it will be put to a vote with several details still to be thrashed out.
</item 5>

<item 6>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
5 Ratcliffe already owns Nice in France and Lausanne in Switzerland Credit: AFP

Initially it was hoped the bidding process would be concluded in time for the summer transfer window, but the longer it dragged on, prompting several
large-scale protests from United fans, the more apparent it became that Joel and Avram were reluctant sellers, as revealed by talkSPORT in March.

Sheikh Jassim’s public statement promising to restore United ‘to former glories’ was also seen as thinly-veiled swipe at the Glazers, which did not help his
negotiating hand.

The Qatari banker’s patience finally snapped and his withdrawal from the bidding process at the end of last week has left the pathway clear for Ratcliffe to
complete what he has always viewed as his dream purchase.

WHAT ARE THE NEXT STEPS FOR RATCLIFFE?

talkSPORT understands Ratcliffe, having originally wanted a much larger up-front stake, has an agreement in principle to buy a 25 per cent share in United.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> France
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> France
- There’s something going on with AI startups in France -MENTIONS-> France
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> France
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> France

Entities: France, Manchester United, Switzerland, talkSPORT, Sir Jim Ratcliffe, Joel, Avram, Nice, Lausanne, Sheikh Jassim, the Glazers, United fans, bidding
process, large-scale protests, Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote
</item 6>

<item 7>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
In November, it will have been a year since the deeply unpopular Glazer family left Manchester United fans rejoicing by announcing they were open to offers for
England’s most iconic football club.

Eleven months on and the saga that, at times has resembled an X-Factor style judges houses audition, may finally be coming to an end as British petrochemicals
billionaire Sir Jim Ratcliffe closes in on a partial takeover.

5 Sir Jim Ratcliffe has an offer on the table for 25 per cent stake of Manchester United Credit: AFP

Ratcliffe’s £1.3billion offer for a 25 per stake in the club he has supported since childhood will be high on the agenda when United’s board of directors meet
on Thursday, but how did we get to this point, what happens next and what could the future look like at Old Trafford?

Graph facts:
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> AFP
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> AFP
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> AFP
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> AFP
- West Indies clinch T20 series after Shai Hope edges hosts past England total -MENTIONS-> AFP

Entities: AFP, Manchester United, England, Glazer family, Old Trafford, talkSPORT, Sir Jim Ratcliffe, Partial takeover of Manchester United, United’s board of
directors meeting, Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote
</item 7>

<item 8>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
As and when it does get as far as a vote, Ratcliffe can be confident it will be ratified seeing as half of the 12-strong board are members of the Glazer family
and in the event of a split vote, Joel and Avram as co-chairmen will have the deciding say.

Even once the final deal is agreed, it is likely to take at least a couple of months for Ratcliffe to pass the Premier League’s owners and directors test, which
has been made more stringent since Portsmouth became the first top-flight side to go into administration in 2010.

5 Ratcliffe, one of Britain's wealthiest men, previously tried to buy Chelsea

WHAT DOES THE FUTURE LOOK LIKE UNDER RATCLIFFE’S INEOS GROUP?

Ratcliffe is demanding an executive role and overall control of day-to-day football matters and plans to appoint his own sporting director, with former
Tottenham and Southampton transfer guru Paul Mitchell one of the main names in the frame.

Graph facts:
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Premier League
- Everton players and fans will be fired up by points penalty, asserts Sean Dyche -MENTIONS-> Premier League
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Premier League
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Premier League
- Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line -MENTIONS-> Premier League

Entities: Premier League, Portsmouth, Southampton, Chelsea, Glazer family, Paul Mitchell, Man United, talkSPORT, Sir Jim Ratcliffe, Joel, Avram, Inside the Man
United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote, INEOS Group, Man United takeover, board vote
</item 8>

<item 9>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
All three players left Nice this summer. Newcastle’s sporting director, Dan Ashworth, and Paul Mitchell, Monaco’s sporting director until March, are among those
mentioned in dispatches as a potential Murtough replacement.

Can any CEO make United a best-in-class elite club again?

Blanc, who has an MBA from Harvard, is billed as the “Lionel Messi of business” by Nice’s chief executive, Fabrice Bocquet. At 60 years old Blanc has been the
Juventus CEO and president, CEO at Paris Saint‑Germain and French Tennis, and the executive overseeing the Tour de France. But once he or whoever becomes
Arnold’s successor is in place, it will dawn rapidly that United are a one-off hydra-headed beast of a club which is treacherously difficult to control. And
there is this, too: across town Manchester City have accrued six championships since United’s last in 2013 and boast a commercial operation that leaves
Ratcliffe’s new concern light years behind.

Graph facts:
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> transfer policy
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Aaron Ramsey
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Jean-Claude Blanc
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Ineos Sport
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Kasper Schmeichel

Entities: Murtough, From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray, Dan Ashworth, Blanc, Harvard, Fabrice Bocquet, Juventus,
French Tennis, Tour de France, Manchester City, Manchester United, Paul Mitchell, Monaco, Arnold, Paris Saint-Germain
</item 9>

<item 10>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Photograph: Michael Regan/Getty Images

Ensuring a working relationship with the Glazers

For paying an overpriced £1.3bn for his 25% Ratcliffe obtains a major element of control but still the Glazers remain majority owners. The 71-year-old oversees
football policy but these do not exist in a vacuum given that the commercial side affects this and vice versa. Ratcliffe accepts Manchester United have an
impressive revenue-based model that makes the club akin to a cash machine but he is not enamoured of how the money generated is invested – primarily in the
squad. What happens then, for example, if Ten Hag wants a new midfield ace, Ratcliffe agrees but the Glazers cite a downturn in income and thus seek to block
the signing?

Graph facts:
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Jean-Claude Blanc
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Ineos Sport
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> transfer policy
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Murtough
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Aaron Ramsey

Entities: From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray, Michael Regan, Getty Images, Manchester United, Ratcliffe, The
Guardian, Glazers, Ten Hag
</item 10>
=== END ITEM id=b1ff85463894f4dd ===

=== ITEM id=3bea8901e4bdfd7e ===
Gold sentences and facts:
Sentence 0: Not only has he produced at a high level, but his play has been instrumental in the Vikings turning things around after a 1-4 start.
  Fact 0.0: He has produced at a high level.
  Fact 0.1: His play has been instrumental in the Vikings turning things around.
  Fact 0.2: The Vikings had a 1-4 start.
Sentence 1: But they've also seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero completions of 20-plus yards from tight ends).
  Fact 1.0: The team under discussion has seen the fourth-lowest ADOT from tight ends.
  Fact 1.1: The ADOT from tight ends is 5.0 yards per target.
  Fact 1.2: Teams seem to know they will not be able to succeed on downfield shots against the team under discussion.
  Fact 1.3: The team under discussion has had zero completions of 20-plus yards from tight ends.

Retrieved items:
<item 1>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The Bears defense comes up big with a 4th down stop



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/yndUqwgIbf — NFL (@NFL) November 28, 2023

9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.

9:46 p.m. — Alexander Mattison has drawn the ire of Vikings fans all season, but he just got Minnesota's second half of to a booming start with a 21-yard run.

End of first half: Bears 3, Vikings 3

9:32 p.m. FIELD GOAL — Greg Joseph drills it from 34 yards out, and this mess of a game is all tied at three entering the half.

9:30 p.m.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Greg Joseph, Vikings, Alexander Mattison, ESPN, Dobbs, Sweat, Brisker, Bears vs. Vikings game, Bears vs. Vikings live score, updates, highlights
from NFL 'Monday Night Football' game, Sporting News, NFLPlus, Bears, ABC
</item 1>

<item 2>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
— Dobbs is nearly intercepted by Jaylon Johnson (again) on third down. The last turnover wasn't Dobbs' fault, but this one should've been a pick. The Bears'
defense is all over the Vikings tonight, and Minnesota can't get out of its own way.

9:13 p.m. — Addison, trying to make up for the drop that turned into an interception, looks like he goes for 17 yards, but replay shows his knee was clearly
down. It becomes a much smaller 4-yard gain.

9:09 p.m. — Fields comes under pressure on third down and misses D.J. Moore. He's starting to miss these receivers after a hot start. Somehow, the Vikings will
have a chance to tie this game or take a lead before halftime.

9:04 p.m. — Dobbs is picked off again!

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Vikings, Fields, Dobbs, Jaylon Johnson, Bears vs. Vikings 'Monday Night Football' game, Addison, Bears vs. Vikings live score, updates,
highlights from NFL 'Monday Night Football' game, Sporting News, D.J. Moore, Bears
</item 2>

<item 3>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
When it mattered most, though, Fields was poised and sealed the Bears' first win over an NFC North opponent since 2021.

While Chicago's hopes are still extraordinarily distant at 4-8, the Vikings saw their playoff chances take a hit on Monday night. Minnesota (6-6) dropped to the
No. 7 seed in the NFC with the loss, with the Packers, Rams and Saints all just a half-game behind in the wild-card race. While a win would have put the Vikings
in position to compete for the division crown down the stretch, just holding onto a playoff spot will now be a battle for Kevin O'Connell's squad.

MORE: Latest updates on Justin Jefferson's return date

The Sporting News tracked live updates from the "Monday Night Football" game between the Bears and Vikings. Check out all of the key moments you might have
missed.

Bears vs.

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Vikings
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Vikings
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- Vikings -MENTIONS-> Minn.

Entities: Vikings, Saints, Packers, Kevin O'Connell, Fields, playoff chances, wild-card race, division crown, Monday Night Football game, Bears vs. Vikings live
score, updates, highlights from NFL 'Monday Night Football' game, Sporting News, Justin Jefferson, Rams, NFC North, Bears
</item 3>

<item 4>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
9:30 p.m. — The Bears' defense has suddenly morphed back into an elite unit on these last two plays, forcing intentional grounding by Dobbs and pressuring him
heavily on the following play. It's 3rd & 24 for the Vikings, who might just settle for a gain of a few yards to set up the field goal.

9:27 p.m. — Another first down by K.J. Osborn drives the Vikings into Bears territory, and the Vikings suddenly find themselves 13 yards from the end zone after
Kyler Gordon is called for interfering with T.J. Hockenson. The Vikings have a chance to take the lead with 38 seconds on the clock.

9:25 p.m. — What an effort by Brandon Powell to turn around and make a 28-yard grab while falling to the ground. That's the biggest offensive play of the night
for the Vikings.

A perfect catch from Brandon Powell on a perfect pass

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Vikings, Brandon Powell, Kyler Gordon, Dobbs, Bears vs. Vikings NFL 'Monday Night Football' game, Bears vs. Vikings live score, updates,
highlights from NFL 'Monday Night Football' game, K.J. Osborn, Sporting News, Bears, T.J. Hockenson
</item 4>

<item 5>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
In an effort not to overwork their backs, who won't have it easy against the Vikings, Kmet could handle more short-area throws than normal. His ADOT over the
past two games has been 7.64, which isn't far from his season average of 6.69.

VIKINGS: Allow the fewest yards per catch to rival tigh ends (5.9 yards), even keeping a lid on Travis Kelce last week (6.7 yards per catch). But they've also
seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero
completions of 20-plus yards from tight ends). These numbers are tough but if Kmet gets the target volume we've seen him have over the past couple of weeks he
can still put up palatable numbers. And, tight ends have scored three times in the Vikings past three games.

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Vikings
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Vikings
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- Vikings -MENTIONS-> Minn.

Entities: Vikings, Travis Kelce, CBSSports.com, NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, Kmet
</item 5>

<item 6>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
And Edwards follows it up with the Bears 3rd INT of the game ‼️



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/2d4C1s2O09 — NFL (@NFL) November 28, 2023

10:06 p.m. — After a hold by Jaylon Johnson gifts the Vikings a first down, they still can't capitalize. Ty Chandler can't make T.J. Edwards miss on third down,
so Minnesota will go for it on fourth down.

10:02 p.m. — The running game is working, the passing game is not for the Vikings. Mattison has 19 yards three plays into this drive for Minnesota.

Bears 6, Vikings 3

9:58 p.m.

Graph facts:
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL 'Monday Night Football' game
- Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL 'Monday Night Football' game
- Garoppolo -PARTICIPATED_IN-> NFL 'Monday Night Football' game
- Alex Anzalone -PARTICIPATED_IN-> NFL 'Monday Night Football' game
- Alim McNeill -PARTICIPATED_IN-> NFL 'Monday Night Football' game

Entities: NFL 'Monday Night Football' game, NFL, Ty Chandler, Vikings, ESPN, Jaylon Johnson, T.J. Edwards, Mattison, Bears vs. Vikings live score, updates,
highlights from NFL 'Monday Night Football' game, Sporting News, NFLPlus, Bears, ABC
</item 6>

<item 7>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:00 p.m. — Will the Josh Dobbs show continue tonight? He’s had good command of the Vikings’ offense in three games since a trade out of Arizona, flashing an
especially strong connection with T.J. Hockenson.

7:45 p.m. — It’s 15 degrees in Minnesota with a feels like of 2, so the roof will come in handy tonight at U.S. Bank Stadium.

7:25 p.m. — Vikings legend Adrian Peterson is in the building to watch his former team.

7:17 p.m. — The Bears enter tonight with the No. 1 and No. 4 picks in the 2024 NFL Draft, and they'll remain in those slots with or without a loss thanks to the
strength of schedule tiebreaker.

7:10 p.m. — Heading into Week 13, tonight's game is the difference between the No. 6 and No. 7 seed for the Vikings.

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Vikings
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Vikings
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Vikings
- Vikings -MENTIONS-> Minn.

Entities: Vikings, Minnesota, Josh Dobbs, Bears vs. Vikings game, Adrian Peterson, Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night
Football' game, U.S. Bank Stadium, Sporting News, Arizona, 2024 NFL Draft, Bears, T.J. Hockenson
</item 7>

<item 8>
Graph query row:
article: NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game
relationship: PUBLISHED_BY
connected_entity: CBSSports.com
</item 8>

<item 9>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Total disaster. In a gamescript that actually gave Minnesota a chance to -- gasp! -- run the ball, Kirk Cousins barely threw the ball and Addison
went catchless on one target. There was also a major reduction in snaps played, down to 58%. His season-low was 56% from Week 1.

MATCHUP: With the Vikings capable of running the ball, the offense went with heavier formations. Backup tight end Josh Oliver played about as many snaps as he
did in Week 1 and third tight end Johnny Mundt saw a season-high 17% of snaps. Week 4 was the first time all year the Vikings DIDN'T throw the ball on at least
69% of their snaps.

THIRD MAN: Last week did affirm that Addison may actually be the No. 3 wide receiver in the offense behind Justin Jefferson and K.J.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Josh Oliver, Vikings, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, Johnny Mundt, K.J.,
CBSSports.com, Addison, fantasy football, Justin Jefferson, Kirk Cousins
</item 9>

<item 10>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
No one has scored a touchdown yet tonight.

10:40 p.m. — Addison's catch stands, and an Alexander Mattison run has the Vikings at the edge of field goal range. They want more than three points, but it's
early enough that they absolutely could settle for a field goal if it got to that point.

10:37 p.m. — Matt Eberflus throws the challenge flag on a nice grab by Addison to give the Vikings a first down. This one looks like it will be tough to
overturn...

10:34 p.m. — Wow, Jordan Addison would've had a walk-in touchdown, but Dobbs' throw takes him out of bounds. Addison couldn't have been more wide open.

10:33 p.m. — Brandon Powell has been the Vikings' biggest playmaker at two key points tonight. He goes 18 yards here to put Minnesota near midfield.

10:29 p.m.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Vikings, Brandon Powell, Alexander Mattison, Jordan Addison, Matt Eberflus, Dobbs, Bears vs. Vikings game, Bears vs. Vikings live score, updates,
highlights from NFL 'Monday Night Football' game, Sporting News, Bears
</item 10>
=== END ITEM id=3bea8901e4bdfd7e ===

=== ITEM id=0f7523852e7a13be ===
Gold sentences and facts:
Sentence 0: "And if the Braves let this be the story of their series, that's on them.
  Fact 0.0: A quoted speaker says that if the Braves let this be the story of their series, that is on the Braves.
Sentence 1: But while Nocturne could have easily settled and coasted on classic Castlevania vibes, the new show steps its game up considerably by using the franchise’s iconic lore to challenge your understanding of what it truly means to fight for freedom, equality, and fraternity.
  Fact 1.0: Nocturne could have easily settled and coasted on classic Castlevania vibes.
  Fact 1.1: The new Nocturne show steps its game up considerably.
  Fact 1.2: The new Nocturne show uses the Castlevania franchise's iconic lore.
  Fact 1.3: The new Nocturne show uses the franchise's iconic lore to challenge the viewer's understanding of what it truly means to fight for freedom, equality, and fraternity.

Retrieved items:
<item 1>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
But in all the bloodshed, Nocturne still finds the time to make its characters (and by extension, you) think about what it means to slaughter technically
innocent people for a genuinely noble cause. That beat isn’t entirely original, and Nocturne’s mindful about leaning on it too hard as it begins to weave
Annette and Edouard’s stories together with Richter and Maria’s in France. Still, it creates a resonance between the show’s action and the concepts it’s
tackling that makes the season as a whole work beautifully.

Image: Netflix

Occasionally, there are moments where the fluid dreaminess of Nocturne’s overall art direction butts up against the show’s application of 3D animation to make
certain pivotal moments pop. It can be jarring when traditionally animated characters who were just popping in and out of view using tricks of the eye suddenly
turn to the camera with deeper faces accented by flat shadows that weren’t there a second before.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> France
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> France
- There’s something going on with AI startups in France -MENTIONS-> France
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> France
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> France

Entities: France, Netflix, Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice, Castlevania: Nocturne, Annette, Edouard,
Richter, Maria, 3D animation, The Verge
</item 1>

<item 2>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
But while Nocturne could have easily settled and coasted on classic Castlevania vibes, the new show steps its game up considerably by using the franchise’s
iconic lore to challenge your understanding of what it truly means to fight for freedom, equality, and fraternity.

Set in the late 18th century during the thick of the French Revolution, Castlevania: Nocturne picks up hundreds of years after the events of Netflix’s first
Castlevania series as demon slayer Richter Belmont (Edward Bluemel) makes his way to France in search of vengeance and vampires. Like his distant holy
whip-wielding ancestor Trevor, Richter’s innate talents for magic make him perfectly suited for getting into the family business, and being raised by his
formidable warrior mother Julia (Sophie Skelton) instills in him a profound sense of pride in the work they do.

Image: Netflix

Together, Richter’s raw potential and Julia’s tutelage make him uniquely powerful even from a young age.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> France
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> France
- There’s something going on with AI startups in France -MENTIONS-> France
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> France
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> France

Entities: France, Netflix, Edward Bluemel, Castlevania, Trevor, Richter Belmont, French Revolution, Castlevania: Nocturne is a revolutionary epic that does the
franchise’s best games justice, Julia, Sophie Skelton, vampires, freedom, equality, fraternity, Castlevania: Nocturne
</item 2>

<item 3>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
Part of / The best entertainment of 2023

Netflix’s first Castlevania animated series excelled by balancing its quiet, beautifully performed moments of human drama with explosive and artful action set
pieces dazzling enough to evoke the feeling of playing Konami’s classic side-scroller for the first time. Despite the first show’s strengths, its regular issues
with pacing and effectively cultivating characters from season to season made the idea of Netflix keeping the vampiric daddy issues ball rolling with
Castlevania: Nocturne — a new spinoff series from executive producer Clive Bradley and co-directors Sam Deats and Adam Deats — a dubious one.

Nocturne both looks and feels a lot like its predecessor in terms of its focus on a tortured scion waging war against machiavellian, bloodsucking ghouls who
fancy themselves undead gods who walk among mortal livestock.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Netflix
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Netflix
- The future of DC movies in 2022 and beyond -MENTIONS-> Netflix
- Fans react to most ‘heartbreaking’ scene in The Crown -MENTIONS-> Netflix
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Netflix

Entities: Netflix, Clive Bradley, Konami, The best entertainment of 2023, Castlevania animated series, Sam Deats, Adam Deats, Castlevania: Nocturne is a
revolutionary epic that does the franchise’s best games justice, Castlevania: Nocturne, The Verge
</item 3>

<item 4>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
She said that reporters abstaining from writing on his comments would be akin to "protecting players from themselves."

She also went on to say that what Arcia said wasn't a slur or hateful, and added the only reason anyone was concerned about it was because "they know it would
motivate Harper to beat [the Braves]. Which implies the Braves do not feel they can handle a motivated Bryce Harper and are willing to suggest that is why they
lost the other day."

"I didn't say something sooner because I thought this was overblown," Janes continued. "And frankly, I thought it wouldn't help. But I am genuinely disgusted by
the way some media members are acting like a reporter was in the wrong for this. He wasn't.

"And if the Braves let this be the story of their series, that's on them. Win and it goes away. Period."

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Braves
- MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency -MENTIONS-> Braves
- Top 25 MLB offseason trade candidates -MENTIONS-> Braves
- MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings -MENTIONS-> Braves
- Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges -MENTIONS-> Braves

Entities: Braves, Sporting News, Bryce Harper, Orlando Arcia, Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash,
Alanna Rizzo rant, Janes, Arcia's comments, slur or hateful, Alanna Rizzo
</item 4>

<item 5>
Graph query row:
article: Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant
publication: Sporting News
entities: ['Blue Jays', 'NLDS', 'NLCS', 'Awful Announcing', 'Harris', 'MLB Network', 'Braves', 'Atlanta', 'Philadelphia', 'FOX Sports', 'X', 'Cincinnati', 'The
Washington Post', 'Bryce Harper', 'Orlando Arcia', 'Atlanta Braves', 'Cespedes Family BBQ', 'Michael Harris', 'Game 2 of the NLDS', 'controversy', 'backlash',
"Atlanta's 5-4 win", '10-2 Game 3 win', 'Kevin Gausman', 'Mintz', 'NLDS Game 2', 'High Heat', 'reporting the comments', 'Bryce Harper Atta Boy controversy',
'Game 2', 'J.T. Realmuto', 'Nick Castellanos', 'Austin Riley', 'Matt Olson', 'Kevin Pillar', 'Chadwick Tromp', 'Chelsea Janes', 'Janes', 'baseball journalism',
'clubhouse reporting', "Arcia's comments", 'slur or hateful', 'Jake Mintz', 'Baseball Writers Association of American', 'MLBN', 'Alanna Rizzo', 'BBWAA', 'Game
4', 'Phillies', 'Game 3']
passages: ['Bryce Harper \'Atta Boy\' controversy, explained: Why Orlando Arcia\'s comments led to backlash, Alanna Rizzo rant (Sporting News,
2023-10-13T21:58:11+00:00)\nIt had been reported Monday night when the Braves beat the Phillies in Game 2 of the NLDS, ending on a critical base-running play
involving Bryce Harper, that Atlanta shortstop Orlando Arcia taunted about the Phillies\' star in the clubhouse after the game.\n\nThat report wound up
generating plenty of controversy.\n\nFox Sports\' Jake Mintz, one of the two brothers that runs the popular Cespedes Family BBQ account, reported that Arcia had
been celebrating after Atlanta\'s 5-4 win by saying "Atta boy Harper" in the Braves\' clubhouse, jeering at the fact that Harper was doubled off on the
basepaths after an outstanding catch by Michael Harris in center field and an impressive relay to get Harper at first to end the game.\n\nThe comments wound up
coming back to bite the Braves two days later, with Harper homering twice — staring down Arcia after each — as he powered the Phillies to the 10-2 Game 3 win.',
'Bryce Harper \'Atta Boy\' controversy, explained: Why Orlando Arcia\'s comments led to backlash, Alanna Rizzo rant (Sporting News,
2023-10-13T21:58:11+00:00)\nWhen the Phillies beat the Braves in Game 4 to eliminate Atlanta and advance to the NLCS, players in the Philadelphia clubhouse
celebrated by repeating, "Atta boy Harper."\n\nWhile the comments became a spark for the Phillies, it also became a subject of controversy. Players, including
the Blue Jays\' Kevin Gausman, decried the situation as a mistake by Mintz to report the comments, which were said out loud in the locker room rather than said
in an interview setting. Others defended the situation, saying that anything in the clubhouse is open to be reported on.\n\nMORE: Why Arcia\'s comments
motivated Harper after NLDS Game 2\n\nBut perhaps the biggest backlash came from MLB Network\'s Alanna Rizzo, who blasted Mintz on "High Heat" for reporting on
the incident. Following a BBWAA statement, Rizzo both publicly and privately apologized to Mintz for her reaction.\n\nHere\'s what you need to know about the
controversy.', "Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News,
2023-10-13T21:58:11+00:00)\nHere's what you need to know about the controversy.\n\nBryce Harper Atta Boy controversy, explained\n\nIt all begins with the play.
In Game 2, Harper led off the top of the ninth in a one-run game with a walk. J.T. Realmuto flew out to center for the first out, but on the next play, Nick
Castellanos lifted a deep fly into right-center. It appeared it would be extra bases, and if Harper took off, he'd be able to score the tying run on the
play.\n\nThe problem for the Phillies was that Harris made a leaping grab up against the wall to rob Castellanos of the hit. Harper had well since rounded
second on his way to third, but the throw from Harris missed both cut-off men by second. Austin Riley, however, ran in and scooped the ball and made an
impressive throw to Matt Olson to nab Harper and end the game.", 'Bryce Harper \'Atta Boy\' controversy, explained: Why Orlando Arcia\'s comments led to
backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)\nAfter the game, Mintz wrote a well-reported story on how the Braves pulled off the
improbable comeback, coming back from down four runs at one point, to win the game, and deep in the article, wrote the following:\n\nKevin Pillar\'s young son,
sporting a custom "Money Mike" headband, waddled over to his favorite center fielder for a fist bump. Third-string catcher Chadwick Tromp strutted by Harris\'
locker offering a congratulatory "Mike is him!" All while Braves shortstop Orlando Arcia cackled emphatically about Harper\'s misfortune, bellowing "ha-ha,
atta-boy, Harper!" repeatedly as reporters circled the room.\n\nAs Mintz pointed out, he was not the only one to allude to the celebration from Arcia. The
Washington Post\'s Chelsea Janes also wrote about it, saying that one of Harris\' teammates was yelling, "Atta boy Harper."', 'Bryce Harper \'Atta Boy\'
controversy, explained: Why Orlando Arcia\'s comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)\nThe only difference was
that Janes did not name the player.\n\nThe reporting of Arcia\'s comments proved to be divisive across the baseball community. Among the most prominent to call
out Mintz was Rizzo. The "High Heat" co-host said she was irritated by "bloggers or podcasters" going into the clubhouse and "ruining it for the rest of us"
when "some jackoff comes in at the end of the season that gets a credential, God only knows why." She called the clubhouse "a sacred place" and defended the
players who were upset by the reporting of Arcia\'s comments, before she called Mintz out by name.\n\nMORE: Every time Nick Castellanos has interrupted a somber
moment with a home run\n\n"First of all, this guy, Jake Mintz, that\'s not even a reporter," Rizzo said. "That\'s taking away from true reporters and true
journalists.', 'Bryce Harper \'Atta Boy\' controversy, explained: Why Orlando Arcia\'s comments led to backlash, Alanna Rizzo rant (Sporting News,
2023-10-13T21:58:11+00:00)\n"That\'s taking away from true reporters and true journalists. Where were you on the random Tuesday in April in Cincinnati when this
team was playing in Cincinnati? Where were you on a rain delay, and you have to sit in the press box for all of this time? It\'s ridiculous."\n\n"And then some
jackoff comes in at the end of the season that gets a credential, God only knows why. And the clubhouse is a sacred space." Alanna Rizzo went off on Jake
Mintz\'s clubhouse reporting on MLBN\'s "High Heat" Thursday. pic.twitter.com/6dooqeLPnh — Awful Announcing (@awfulannouncing) October 12, 2023\n\nRizzo\'s
comments immediately drew heat from the baseball journalism world, including from Janes, who posted a thread defending Mintz. Janes reported that cameras and
recorders were recording, and that Arcia yelled the comments while they were recording and with more than a dozen reporters around.', 'Bryce Harper \'Atta Boy\'
controversy, explained: Why Orlando Arcia\'s comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)\nShe said that reporters
abstaining from writing on his comments would be akin to "protecting players from themselves."\n\nShe also went on to say that what Arcia said wasn\'t a slur or
hateful, and added the only reason anyone was concerned about it was because "they know it would motivate Harper to beat [the Braves]. Which implies the Braves
do not feel they can handle a motivated Bryce Harper and are willing to suggest that is why they lost the other day."\n\n"I didn\'t say something sooner because
I thought this was overblown," Janes continued. "And frankly, I thought it wouldn\'t help. But I am genuinely disgusted by the way some media members are acting
like a reporter was in the wrong for this. He wasn\'t.\n\n"And if the Braves let this be the story of their series, that\'s on them. Win and it goes away.
Period."', 'Bryce Harper \'Atta Boy\' controversy, explained: Why Orlando Arcia\'s comments led to backlash, Alanna Rizzo rant (Sporting News,
2023-10-13T21:58:11+00:00)\nWin and it goes away. Period."\n\nThe incident elicited a response from the Baseball Writers Association of American (BBWAA), which
defended Mintz, a BBWAA member.\n\n"For the record, Jake is a BBWAA member in good standing and has earned the right to be credentialed through his hard work
and quality baseball coverage," the BBWAA wrote in the statement. "To assert otherwise, in vulgar terms, is both unprofessional and unacceptable. The BBWAA is
deeply troubled that the league\'s own network would permit the disparaging of one of our members in this fashion. Scrutinizing our work is part of the
territory but comments such as these should have no place on MLBN."\n\nFollowing the backlash to her comments, Rizzo apologized on "High Heat" to Mintz and the
BBWAA. She said her comments were "completely inappropriate, it was inaccurate and it was beyond unprofessional.', 'Bryce Harper \'Atta Boy\' controversy,
explained: Why Orlando Arcia\'s comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)\nI got caught up in the heat of the
moment."\n\n"What I said was completely off base and I\'ve had an interesting 24 hours and I\'ve learned a lot more and I apologize to Jake. I talked to him
already, I apologized to him privately and I also apologize to him publicly," Rizzo said. "They do a good job in the clubhouse and I was completely inaccurate
in my assessment of what went down there."\n\nAlanna Rizzo gave "a very heartfelt apology to Jake Mintz and the entire BBWAA" pic.twitter.com/rT9tjSWLY3 — Awful
Announcing (@awfulannouncing) October 13, 2023\n\nMintz, though calling her initial comments "incredibly hurtful/uncalled for," wrote on X saying that he was
"glad she apologized," and asked people not to toss hate to her.']
</item 5>

<item 6>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
The only difference was that Janes did not name the player.

The reporting of Arcia's comments proved to be divisive across the baseball community. Among the most prominent to call out Mintz was Rizzo. The "High Heat"
co-host said she was irritated by "bloggers or podcasters" going into the clubhouse and "ruining it for the rest of us" when "some jackoff comes in at the end
of the season that gets a credential, God only knows why." She called the clubhouse "a sacred place" and defended the players who were upset by the reporting of
Arcia's comments, before she called Mintz out by name.

MORE: Every time Nick Castellanos has interrupted a somber moment with a home run

"First of all, this guy, Jake Mintz, that's not even a reporter," Rizzo said. "That's taking away from true reporters and true journalists.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, Bryce Harper, Orlando Arcia, Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo
rant, Nick Castellanos, High Heat, Janes, Jake Mintz, Alanna Rizzo
</item 6>

<item 7>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
It’s a war between undead ghouls like Drolta Tzuentes (Elarica Johnson) manipulating mortals from the shadows, selfish humans like a power-hungry Abbot
(​​Richard Dormer), and agitators like summoner Maria (Pixie) who want to dismantle the entire oppressive system of power that makes France a formidable global
presence.

Image: Netflix

Narratively, Nocturne’s focus on another brooding Belmont and its depiction of late-18th-century France as a country overrun by literally vampiric aristocrats
drunk on the promise of a new messiah make the new series feel true to the essence of Castlevania. What makes Nocturne’s deep dive into this alternate history a
worthwhile watch, though, is the way it presents France not just as a crumbling monarchy in the throes of a heroic social uprising but also as a colonial nation
whose pivot to democracy at home can’t be divorced from its history of committing unfathomable atrocities abroad.
</item 7>

<item 8>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
But despite all of Julia’s skill with her family’s deadly heirloom, Castlevania: Nocturne’s world is one teeming with vampires even centuries after Dracula has
seemingly been destroyed, and the series opens at a devastating moment in Richter’s past that leaves him orphaned and unable to access his magic.

Though Nocturne takes place in the same continuity as the previous Castlevania show, it also uses major pieces of lore from Konami’s Castlevania: Rondo of Blood
and Symphony of the Night games to tell a riveting story about the inherent hypocrisies baked into the historic record of colonial nations embracing democratic
revolution.

Here, the French Revolution isn’t just a transformative period of social unrest between moneyed elites and impoverished peasants fed up with their subjugation.

Graph facts:
- The best new TV series on every streaming service -MENTIONS-> French Revolution
- The Halloween Countdown: 31 days of horror to watch -MENTIONS-> Dracula
- The best new TV series on every streaming service -MENTIONS-> Dracula
- The best new TV series on every streaming service -MENTIONS-> Konami
- Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice -MENTIONS-> France

Entities: Konami, Dracula, French Revolution, Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice, Julia, vampires,
Castlevania: Nocturne, Castlevania, Castlevania: Rondo of Blood, Symphony of the Night, democratic revolution, colonial nations, Richter, The Verge
</item 8>

<item 9>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
Nocturne reworks a number of Castlevania’s classic supporting characters like Maria — who fights using reimagined versions of the Four Sacred Beasts — and
empathic singer Edouard (Sydney James Harcourt) to brilliant effect. But out of all the show’s old but new faces, ferrokinetic Vodoun practitioner Annette
(Thuso Mbedu) stands out because of how strongly the show introduces her as a woman whose Haitian identity is what makes her able to understand the situation
over in France with an exacting kind of clarity.

It’s through Annette and the show’s exploration of Afro Caribbean syncretism that Nocturne expands the boundaries of Castlevania’s magical mythology, which is
touched on heavily throughout the season’s eight episodes. Small nods to the video games and gnarly beheadings abound, particularly in the Powerhouse
Animation-produced show’s bigger hyperkinetic brawls between its heroes and hordes of horrific creatures.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> France
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> France
- There’s something going on with AI startups in France -MENTIONS-> France
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> France
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> France

Entities: France, Thuso Mbedu, Castlevania, Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice, Nocturne, Four Sacred
Beasts, Sydney James Harcourt, Haitian identity, Afro Caribbean syncretism, Castlevania’s magical mythology, Powerhouse Animation, Annette, Edouard, Maria, The
Verge
</item 9>

<item 10>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
As strong and different as the season starts, by its last two episodes, Nocturne does start to feel a bit like it’s rehashing moments from the first show — if
not in exact form, then in terms of how neatly it brings its plotlines to a close. None of that’s enough to keep the show from hitting all the right notes,
however, and by the time you hit the finale, there’s a solid chance you’ll already be hoping for more.

Graph facts:
- Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice -MENTIONS-> France
- Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice -MENTIONS-> Netflix
- Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice -MENTIONS-> Clive Bradley
- Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice -MENTIONS-> Edward Bluemel
- Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice -MENTIONS-> Thuso Mbedu

Entities: Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice, Castlevania: Nocturne, The Verge
</item 10>
=== END ITEM id=0f7523852e7a13be ===
