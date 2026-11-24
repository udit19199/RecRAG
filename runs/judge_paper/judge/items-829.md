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

This file has 4 items.

=== ITEM id=26be7abd23ae5259 ===
Gold sentences and facts:
Sentence 0: Erik ten Hag is a top coach from Ajax but they need a top blueprint not only for now but for the next five years, and work towards that, and sometimes on that journey there are hard moments when you don't get the right results, even though you're doing the right things.
  Fact 0.0: Erik ten Hag is a top coach from Ajax.
  Fact 0.1: The club needs a top blueprint not only for now but for the next five years.
  Fact 0.2: The club should work towards that blueprint.
  Fact 0.3: Sometimes on that journey there are hard moments when the club does not get the right results.
  Fact 0.4: The club can be doing the right things even when it does not get the right results.
Sentence 1: Manchester United are out of Europe altogether after a limp display against Bayern at Old Trafford, where Kingsley Coman sealed a 1-0 win for Thomas Tuchel's side.
  Fact 1.0: Manchester United are out of Europe altogether.
  Fact 1.1: Manchester United put in a limp display against Bayern at Old Trafford.
  Fact 1.2: Kingsley Coman sealed a 1-0 win at Old Trafford.
  Fact 1.3: Kingsley Coman's 1-0 win was for Thomas Tuchel's side.

Retrieved items:
<item 1>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
"We are a long time from the Sir Alex Ferguson era. Erik ten Hag is a top coach from Ajax but they need a top blueprint not only for now but for the next five
years, and work towards that, and sometimes on that journey there are hard moments when you don't get the right results, even though you're doing the right
things.

"You have to know that, over that period, the consistency of the decision-making will yield a good end result. I think we've seen that both Manchester City and
Liverpool, that when you adopt that long-term strategy you can get real consistency and sustainable success.

"I think that's the biggest thing for Manchester United; putting down their identity of what they want to be today, but most crucially what they want to be in
five years' time, and where they want to be, and what they want to look like; from everything from young players, to the style of play, to the whole culture
peace in Manchester United."
</item 1>

<item 2>
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
</item 2>

<item 3>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
5 Joel and Avram Glazer put the Red Devils up for sale in November 2022 Credit: AFP

5 United supporters are desperate to see the Glazers depart Old Trafford Credit: Twitter

The fact the Glazers will remain in situ has not gone down well with supporters, but this is seen as the first step to Ratcliffe eventually removing the
Americans and taking overall control.

The exact timescale as to when that happens remains vague and how much it will cost Ratcliffe overall could depend on United’s sporting and commercial success
in the coming seasons.

Thursday’s board meeting was pre-arranged and while it is expected that Ratcliffe’s proposal will discussed in detail, sources close to United have indicated to
talkSPORT that it is far from guaranteed it will be put to a vote with several details still to be thrashed out.
</item 3>

<item 4>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
No family member has given a press conference or interview during their 18-year tenure. Ratcliffe has arrived on a ticket of, partly, being the local lad made
good (Failsworth, where he was born, is next to Newton Heath, United’s birthplace) so if he proves as silent as the Americans then this will be: a) poor PR; and
b) just plain poor. Football fans can be classed as the most loyal citizens of any society but even the faith of United enthusiasts has been stretched by the
cold relationship created by the Glazers’ stance regarding communication. A tap-in, then, for Ratcliffe is to remedy this by opening and maintaining a regular
line into supporters.

The Glazer family have little relationship with the Manchester United fans – something Ratcliffe would do well to avoid.
</item 4>

<item 5>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
Finally you get to manage the everyday metrics on which you will be judged; team, results, style, energy, messaging. Win things, improve players, create a
coherent and happy internal culture: but do it during an unceasing takeover circus while star players sulk and brief against you, while the ghost of Hamlet’s
father still skulks in the eaves. This felt like the end of one thing, perhaps even the beginning of the end of something else.
</item 5>

<item 6>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Manchester United’s century old stadium needs an upgrade. Photograph: Greig Cowie/Shutterstock

Communications

The Glazers treat a microphone or TV camera as United fans do Liverpool or Manchester City: as a fierce enemy. No family member has given a press conference or
interview during their 18-year tenure.
</item 6>

<item 7>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
As the seconds ticked down at the end of a Champions League season‑ender that felt like a gentle, even quite tender, act of sporting euthanasia, the only noise
inside Old Trafford was the sound of the Bayern Munich fans singing an impressively sustained version of Football’s Coming Home. A little later they sang Is
This A Library? Actually, no, it’s more of a museum. They chanted “Auf Wiedersehen” at the departing members of the home support who, frankly, had done pretty
well to stick it out to that point.

The good news for Erik ten Hag at the end of this low-fi 1-0 defeat is that United’s players didn’t lie down and die at Old Trafford. Instead they stood up and
died, running hard and creating a kind of simulacrum of a functioning elite team.

This was not a group of players in revolt or drained of spirit. They looked like what they are, a muddled creation, stitched together out of parts and off-cuts.
</item 7>

<item 8>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Photograph: Michael Regan/Getty Images

Ensuring a working relationship with the Glazers

For paying an overpriced £1.3bn for his 25% Ratcliffe obtains a major element of control but still the Glazers remain majority owners. The 71-year-old oversees
football policy but these do not exist in a vacuum given that the commercial side affects this and vice versa. Ratcliffe accepts Manchester United have an
impressive revenue-based model that makes the club akin to a cash machine but he is not enamoured of how the money generated is invested – primarily in the
squad. What happens then, for example, if Ten Hag wants a new midfield ace, Ratcliffe agrees but the Glazers cite a downturn in income and thus seek to block
the signing?
</item 8>

<item 9>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
That raises doubts over the future of current sporting director John Murtough and chief executive Richard Arnold, although manager Erik ten Hag is believed to
be on a solid footing despite United’s poor start to the season.

Expect Sir Dave Brailsford, the mastermind behind Team GB’s cycling success at the London 2012 Olympics and director of sport for Ratcliffe’s company INEOS, to
have a prominent role in any reshuffle.

Ratcliffe also has ambitious plans to upgrade Old Trafford and turn it into a 90,000-seat stadium.

He wants the deal ratified in time for the January transfer window, however due to FFP concerns, there is no guarantees United will be able to spend big.

Despite the fact he turns 71 a day before the board meeting, it appears Ratcliffe’s plans centre around long-term stability and success rather than a short-term
turnaround in United’s fortunes.
</item 9>

<item 10>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
As Manchester United's season continues to slip away, the pressure around Dutch boss Erik ten Hag continues to get more palpable.

The latest slip saw the Red Devils crash out of Europe as they fell 1-0 to Bayern Munich on the final day of Champions League group-stage play, finishing bottom
of their four-team group to leave them eliminated from Europe in their worst performance in the competition since 2005.

On top of that, they set the record for most goals against in a Champions League group stage by an English club, with 15 conceded in their six games. Their
defensive record has collapsed as the season goes on: from a Premier League standpoint, they sit eighth in the English top flight, but their -5 goal
differential is the worst of any club in the top 10.

With Jim Ratcliffe having now completed the formalities around his investment into the club, taking control of 25% of his boyhood club after a year of
negotiations, the pressure will be piled on even more.
</item 10>

<item 11>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Old Trafford (and the training ground)

To find an apt shorthand for the lurching ship that is United, end the search at the roof of their stadium where on Manchester’s (many) wet days rain leaks
through. Old Trafford remains a storied venue but as it continues to crumble the tale of disrepair has become a central narrative in its recent history. As
Premier League titles have become scarce (zero in a decade) the tale of neglect of a ground that dates to 1910 is emblematic of the Glazers’ quasi‑absentee
proprietorship. Ratcliffe will invest $300m (£237m) into infrastructure but this is small beer when considering the multibillion-pound facelift or new venue
required. There is also the cramped Carrington base: as on Sir Matt Busby Way, the footprint is large at the training ground but the configuration and
facilities are ageing – badly.
</item 11>

<item 12>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
In the second half United’s back five was André Onana, Aaron Wan-Bissaka, Jonny Evans, Raphaël Varane and Diogo Dalot, an entirely random collection of players
signed, at various times, by Ten Hag, José Mourinho, Ole Gunnar Solskjær and Alex Ferguson. Presumably Daley Thompson, Michael Portillo and Howard from Take
That were busy.

By the end United had five academy players on the pitch, ages ranged between 17 and 35, in outline evidence of internal good health, in reality another symptom
of a machine that wants to function properly but has for so long been made to judder along at half-speed. Ever feel like you’re starting to slip into the past
tense for good? When VAR happens at Old Trafford there isn’t even a VAR screen to say “VAR Check”. The 1990s were a hell of a decade. They still are around
here.
</item 12>

<item 13>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
All three players left Nice this summer. Newcastle’s sporting director, Dan Ashworth, and Paul Mitchell, Monaco’s sporting director until March, are among those
mentioned in dispatches as a potential Murtough replacement.

Can any CEO make United a best-in-class elite club again?

Blanc, who has an MBA from Harvard, is billed as the “Lionel Messi of business” by Nice’s chief executive, Fabrice Bocquet. At 60 years old Blanc has been the
Juventus CEO and president, CEO at Paris Saint‑Germain and French Tennis, and the executive overseeing the Tour de France. But once he or whoever becomes
Arnold’s successor is in place, it will dawn rapidly that United are a one-off hydra-headed beast of a club which is treacherously difficult to control. And
there is this, too: across town Manchester City have accrued six championships since United’s last in 2013 and boast a commercial operation that leaves
Ratcliffe’s new concern light years behind.
</item 13>

<item 14>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
"So I think there's a lot of question marks in the performances and the way they're going about things at the moment but balance for me is one of the key
aspects that they've got to really find.

"We mentioned in commentary, scars from throwing away leads, those are hard to get away, you've got to really manage games and he mentioned that himself Bruno,
you've got to go out managing games correctly.

"They've got to do that not over one or two or three games, they've got to do that over a period of games now so they can erase those scars that they've created
and they'e grown through bad results and throwing away leads, especially in Europe."
</item 14>

<item 15>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
United have done little learning in recent times, repeating the same failures in Europe’s elite competition. It would not be a disgrace to lose against a good
Galatasaray team, especially given Erik ten Hag is missing key players, but if the worst does happen, everyone will need to come out of the experience stronger.
Regardless of the result, it should be a night of education when it comes to the demands of the Champions League.

Six of United’s squad that failed in 1993 went on to win the competition six years later. “It did galvanise us,” says Mike Phelan, who started in central
midfield 30 years ago. “Winning the Cup Winners’ Cup took us to another level and we felt pretty comfortable with each other and felt we could develop into a
serious team, and I think going into the Champions League was one of those moments where we expected to do well – the experiences along the way were great and
took us to the next level further down the line.
</item 15>

<item 16>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Almost certain to consult Rangnick before any approach from Ratcliffe, Mitchell may be warned off of a club where long-term thinking hasn’t existed.

But, should Ratcliffe get his man, it could signal exactly the change in fortunes he and every supporter has been hoping for at Manchester United.
</item 16>

<item 17>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
The 1990s were a hell of a decade. They still are around here.

'We've lost decisive players': Ten Hag says injuries were key to Champions League exit – video

So much for the good news then. The bad news is that Ten Hag’s United have now compiled arguably the worst performance by any English team in a Champions League
group stage. OK, it’s no Blackburn Rovers. But there are extra marks here for United’s pedigree, for the fact the group really wasn’t much of a group, and for
United managing to finish bottom, shipping 15 goals in the process, more than any other English team ever.

Plus of course there were just so many moments of farce along the way, a six-part saga of chaos, collapse and laughter in the dark. We remember Onana diving out
of the way of the ball in Munich. We remember Onana also diving out of the way of the ball in Istanbul.
</item 17>

<item 18>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
They still have the FA Cup, and the extended pursuit of fourth or even fifth place, although the state of the coefficient means the thing that might keep
Manchester United out of Europe next season could be Manchester United being so bad in Europe this season. Maybe City can revive their fortunes by winning the
Champions League again.

Ten Hag will take the blame of course for the early exit and plenty of mistakes have been made, not least in the players the manager has been allowed to sign.
In reality assessing the job any United manager does with this team involves applying a series of filters. To be United manager is to undertake three concurrent
roles.

First you must manage the past, which is constantly in the room and which skews every act, achievement and expectation. Second you must manage the dysfunction
of the present, which expresses itself in every detail from a leaking stadium roof to the deep ills of a playing squad peopled with squatters and long‑term
missteps.
</item 18>

<item 19>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
After all that pressure, all that tender hope, it looked a bit like a training goal.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

And that was pretty much that. From here United’s season continues to narrow. They can’t win the league now, can’t win the Champions League, can’t win the
Carabao Cup.
</item 19>

<item 20>
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
</item 20>
=== END ITEM id=26be7abd23ae5259 ===

=== ITEM id=7525338227a98e32 ===
Gold sentences and facts:
Sentence 0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 0.0: Wang testified that Wang heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account.
  Fact 0.1: The withdrawals were okay as long as the amount was less than FTX's total trading revenue.
Sentence 1: That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.
  Fact 1.0: The answer to the question ultimately lies with the jury.
  Fact 1.1: The jury was selected on Wednesday morning.
  Fact 1.2: The jury was selected before lawyers for the government.
  Fact 1.3: Bankman-Fried swapped two very different stories of the former crypto mogul's sudden rise.
  Fact 1.4: Bankman-Fried swapped two very different stories of the former crypto mogul's almost instantaneous fall.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power, and influence.
  Fact 2.1: The defense countered that the FTX founder acted in good faith.
  Fact 2.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
</item 1>

<item 2>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
Meanwhile, Bankman-Fried tweeted. “FTX is fine. Assets are fine” and “We don’t invest customer assets even in treasuries.”

Pointing at Bankman-Fried, Rehn said, “This man stole billions of dollars from thousands of people.”

So how was the defense going to follow it up? I was very curious, having learned yesterday that Bankman-Fried had never been offered a plea deal since he and
his attorneys had told the government they wouldn’t negotiate. Surely there would be some manner of evidence, some something, that would have made him so
confident.

There was, instead, a metaphor.

Defense attorney Mark Cohen, with the energy of a patient father telling his obnoxious children a bedtime story, assured us that working at a startup was like
building a plane while flying it, and that FTX the plane had flown right into the perfect storm: the crypto crash.
</item 2>

<item 3>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 3>

<item 4>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
There was a minimum of word salad today. I don’t know if he was more relaxed, or he’d just been more rehearsed, but I will certainly be watching to see if he
suddenly becomes much less coherent when Sassoon gets him for the cross.

Here is the story of FTX, from his point of view.

Bankman-Fried, who informed us he’s “somewhat introverted, naturally,” gave us a rather prolonged tour of his pre-Alameda Research life, which I will skip. In
2017, during a crypto bull run, he started his cryptocurrency trading firm. He knew “basically nothing” about cryptocurrency at the time, he explained, but he
wanted to do arbitrage on it anyway.

Alameda Research was named for Alameda County in California, which was where its first office was set up. As for its name, here’s what Bankman-Fried said on the
stand:

Effectively, we wanted to be under the radar at that point in time.
</item 4>

<item 5>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
“I worried if I didn’t, I would lose touch with the actual concerns of the customers,” he testified. What he did not do was create a risk team, which he is now
characterizing as a “big mistake.”

Risk is an inherent part of a futures exchange, which is even more like a casino than regular cryptocurrency. Not having a risk team, when you are any kind of
financial anything, is certainly a choice. It is especially a choice when you go around telling everyone your crypto exchange is very good and safe.

FTX’s big selling point was its “risk engine,” which was supposed to prevent big losses that would then be spread around all the rest of the customers. But
Bankman-Fried testified that in 2020, the “risk engine was effectively sagging under the weight” of the exchange’s rapid growth. So its time to liquidation went
up — it took minutes to determine which accounts needed to be liquidated.
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 6>

<item 7>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.
</item 7>

<item 8>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Perhaps you are wondering, was the jury also there? Reader, it was.

Bankman-Fried denied he knew about the effectively infinite line of credit Alameda Research received from FTX. This argument was peculiar; essentially my
take-away was that the CEO of a financial company simply didn’t pay attention to finances.

FTX couldn’t get bank accounts right away. Bankman-Fried anticipated it would take a year or two. Rather than wait, he decided to use Alameda as the “payment
provider” for bank transfers. “My understanding at the time was that there were teams managing the process,” he said. “At the time, I wasn’t entirely sure what
was happening.”

Well, sure, understandable! He’s an introvert!

Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the line” to $1 billion

In 2021, FTX was growing to millions of users, with $1 billion of revenue.
</item 8>

<item 9>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.
</item 9>

<item 10>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
It is a thankless task to cross-examine a customer whose money is gone

In November 2022, things went bad for Julliard. He followed Bankman-Fried on Twitter, and read aloud the “FTX is fine. Assets are fine” tweets, along with “FTX
has enough to cover all client holdings. We don’t invest client assets” and a few others, which gave Julliard the impression that his money was there — the
problem might have been technical (anti-spam measures) or regulatory. When he tried to get his money out on November 8th, it was too late. We saw screenshots of
his withdrawal attempts: $20,000 USD and about 4 Bitcoin, which were worth about $20,000 at the time: about $100,000 money, inaccessible.

It is a thankless task to cross-examine a customer whose money is gone, but Cohen tried anyway.
</item 10>

<item 11>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
When he found it, he was “very surprised!”

Besides the two obvious lies Bankman-Fried told on the stand — about Alameda Research’s name and about “allow_negative” — I have been struck by how little he
seems to know about his own companies. Apparently, Singh, Wang, and Ellison were out there just doing whatever their little hearts desired. Because
Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at his crypto trading firm and futures exchange.
</item 11>

<item 12>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Here are the hits:

Bankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though FTX hadn’t been in the business as long as competing
exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish face for crypto. (At one point, Bankman-Fried told a
colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for me to cut my hair. I think it’s important for people
to think I look crazy.”)

Because he was so successful at this kind of public relations, his fall from grace was another mark against an industry that was already roiled by bankruptcies
and scandals. Some additional trouble for the crypto industry is likely to come from one crucial element of the fraud trial — the part where the government must
prove intent.

The first part of proving the government’s case is pretty simple and a little boring: prosecutors must show that certain transactions took place.
</item 12>

<item 13>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 13>

<item 14>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 14>

<item 15>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
When he was evaluating the exchange, he thought the sheer volume of users was important, too — at the time, FTX was among the top three biggest exchanges. Plus,
major VC firms had invested, and “they don’t commit hundreds of millions without doing due diligence, checking the books, the accountancy of the firm, going
through several compliance process[es], so that was a vote of confidence for me,” Juilliard said. (Evidently he had not paid attention to the Elizabeth Holmes
trial.)

He also noted FTX’s glossy ads — featuring Gisele Bündchen, for instance — suggested a very high budget. It wouldn’t make sense to spend that much money unless
FTX had very strong financials, Juilliard figured. He opened an account, transferred in both regular money and cryptocurrency, and used the exchange to execute
his plan: buying Bitcoin to sell back in five to ten years at higher prices.
</item 15>

<item 16>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.
</item 16>

<item 17>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“Sam acted in good faith and took reasonable business measures” is a pretty hard pill to swallow with that in mind.

Man, it’s no good when your defense lawyer has just made you sound worse than the prosecution already did. And while Cohen tried to make the common white-collar
defense argument that Bankman-Fried, as CEO, was simply too busy to oversee what everyone did every day, he just made me more suspicious. That’s why you hire a
risk officer and delegate! That’s the whole point! I could barely even hear Cohen blaming Caroline Ellison and Changpeng “CZ” Zhao for the debacle over the “no
risk officer” ringing in my ears.

Following the defense’s opening statements, things got still worse for Bankman-Fried. The prosecution called its first witness, Marc-Antoine Julliard, whose
money got stuck on FTX. Juilliard, who was born in Paris and lives in London, testified that he trusted FTX because Bankman-Fried came across as a leading
figure of the industry.
</item 17>

<item 18>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
The general sense I got, from the testimony of Bankman-Fried’s former friends, was that things just kept escalating

During Yedidia’s testimony, we were shown a photograph in court: former President Bill Clinton and former Prime Minister Tony Blair onstage in suits next to
Bankman-Fried, in a rumpled T-shirt and cargo shorts. Sure, dressing down is a staple of Silicon Valley startup culture — and a sign of its swagger. The defense
counsel tried to elicit testimony from Yedidia that the cargo shorts were evidence of Bankman-Fried’s humility, but the photograph seemed to signify just the
opposite: that the man in cargo shorts is so important, he doesn’t need to dress up, even to meet former world leaders.

Now, in court, Bankman-Fried is wearing a suit.

In 2019, just months after FTX was founded, Wang oversaw code called “allow_negative” that let Alameda Research avoid being liquidated by FTX’s risk management.
It was turned on in July.
</item 18>

<item 19>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.
</item 19>

<item 20>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 20>
=== END ITEM id=7525338227a98e32 ===

=== ITEM id=ccfcbf48ccf4fa20 ===
Gold sentences and facts:
Sentence 0: The winger came firing out of the gate, but the outlook has changed for the Devils in recent days after losing the superstar center Hughes to injury.
  Fact 0.0: The winger came firing out of the gate.
  Fact 0.1: The outlook has changed for the Devils in recent days.
  Fact 0.2: The Devils lost the superstar center Hughes to injury.
  Fact 0.3: The outlook for the Devils changed after they lost Hughes to injury.
Sentence 1: Having to depend much on his weaker arm, Hamlin stayed in contention for the championship four race until the final race in Martinsville.
  Fact 1.0: Hamlin had to depend much on his weaker arm.
  Fact 1.1: Hamlin stayed in contention for the championship four race.
  Fact 1.2: Hamlin stayed in contention until the final race in Martinsville.

Retrieved items:
<item 1>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning in three limited practices this week, we'll trust that Hockenson is ready to go, especially since he continued to play through the ailment in Week
9 as a big part of the offense with Josh Dobbs under center.

As long as Hockenson is playing, he's not being benched in any fantasy format. The tight end projects as a top option at the position and remains the Vikings'
leader in targets with 80 on the season. Dobbs enjoyed targeting the tight ends in Arizona and continued that trend in his Minnesota debut. The floor for
Hockenson is equivalent to that of a rooftop bar.

WEEK 10 DFS: DK lineup | FD lineup | Best values | Best stacks

Is Curtis Samuel playing Week 10?

UPDATE: Samuel is officially ACTIVE.

Samuel (toe) is expected to play on Sunday when the Commanders take on the Seahawks at 4:25 p.m. ET, according to ESPN's Jeremy Fowler.
</item 1>

<item 2>
“We Sell More F-150s to NASCAR Fans” – Ford Insider Reveals Insane Mustang Stat Amid Big Change (Essentially Sports, 2023-11-06T23:01:17+00:00)
The wisdom of experienced leaders like Roger Penske plays a crucial role in the success of their teams.

“We’re Champions”- Roger Penske Flexes on Rick Hendrick As He Publicly Boasts Team Penske’s Strength With Back-to-Back Titles
</item 2>

<item 3>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
These game plans were conservative enough to signal a potential in-season quarterback change. Denver faces Green Bay and Kansas City over the next two weeks
before heading into its Week 9 bye.

5. You play to win the game, but can you win too quickly? The Giants and Vikings are test cases.

The New York Giants and Minnesota Vikings exceeded expectations under first-year head coaches in 2022. Both reached the playoffs. Both are struggling in their
second seasons, facing difficult questions about the future. Would they have been better off enjoying only moderate success early? That could depend on how well
the football leadership has communicated with ownership.

Every head coach and general manager wants to win as much as possible, but the goal is to build for the long term, which can require time to do things with
sustainability in mind. Bad teams that win right away can reset expectations for fans and ownership, making it less practical to take one step backward in the
short term for a better shot at taking two steps forward down the line.
</item 3>

<item 4>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Tuesday night's NHL national games feature an Original Six matchup at Madison Square Garden in addition to a battle between the team that left and the team that
took its place. The Rangers and Red Wings will face off at the world's most famous arena as both squads have jumped out to hot starts in the new season. Dylan
Larkin is a game-time decision for the Motor City while injuries to the Blueshirts' Adam Fox, Filip Chytil and Igor Shesterkin are making sure some of this
matchup's biggest stars will not be watching in uniform. Regardless, there is still plenty of star power on Broadway in what should feature plenty of offense.

In the late game, the Devils return to Colorado -- the place they left in order to move to New Jersey -- for a showdown with the Avalanche. Again, injuries are
a big part of the storyline here, thanks to Jack Hughes and Nico Hischier out until further notice.
</item 4>

<item 5>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
Stern then pointed out how the media rights revenue plays a key role in sports as he used the MLS as an example.

He said, “You look at MLS. MLS takes a big deal with Apple, almost all of their games behind the Apple TV’s paywall. That was the money they could get and they
had to deliver for their industry. Their industry is expected to be a growing business and one of the most sporting ways to be a growing business in sports is
to have growing media rights revenue.”

While he explained that it was a similar case with NASCAR, he added one more angle because of which this deal possibly materialized. It is no secret that the
cable universe has been declining. As Stern discussed, every quarter, there is a drop in a couple of percentages of subscribers, which is very concerning.

This is one of the reasons why NASCAR did not want to gamble solely on cable users.
</item 5>

<item 6>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 6>

<item 7>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
Discovery, those races are going to be simulcast not just on MAX streaming but also on TNT and TNT is cable, not a free-to-air channel.”

Stern continued explaining how it can come back to haunt. He said, “You have that on the cable subscription. So, the number of races that are going down that
have been on the network, that’s what teams are concerned about because the ratings will probably be lower.”

ADVERTISEMENT Article continues below this ad

With not every fan having access to all platforms, it is a tricky situation both for the fans and NASCAR. However, the sanctioning body has gone ahead and made
its move with the new media rights deal. Now, it remains to be seen how the execution part works out. For now, it looks like NASCAR has secured itself well by
bringing more money to the table.

Kyle Larson Believes NASCAR’s Multi-Billion Dollar Media Rights Deal Is a Step in the Right Direction Despite Fans’ Backlash
</item 7>

<item 8>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 8>

<item 9>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
Mandatory Credit: Gary A. Vasquez-USA TODAY Sports

While his on-track expertise was one of the notable points in 2023, his off-track antics didn’t go unnoticed either. Ross Chastain was clouded in controversy
after a fistfight with Noah Gragson and on-track altercations with Denny Hamlin and Kyle Larson. While his cutthroat racing style appeased the fans, his driving
skills didn’t waver either.

ADVERTISEMENT Article continues below this ad

Chastain scored two wins, ten 1op-5s, and 14 top-tens, while leading 640 laps in 2023. He also led the NASCAR Cup Series standings for seven weeks and had a
career-best 28 lead lap finishes. Overall, Ross Chastain has been one of the most resilient NASCAR drivers of 2023 and deserves a mention among NASCAR’s very
best.
</item 9>

<item 10>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
After running the car full-time in 2023, it was announced Thursday that A.J. Allmendinger would go back to full-time Xfinity Series racing for Kaulig in 2024,
though he will continue to run a partial Cup schedule.

The assumption of many had been that the No. 16 would be taken over by Ty Dillon, but that assumption may have been premature. Weaver shared in a recent thread
on X that information suggesting Dillon had been hired to drive the No. 16 was "incorrect," casting ambiguity on both who will join Daniel Hemric at Kaulig next
season as well as the exact plans of Dillon, who has bounced from team to team since Germain Racing shut down following the 2020 season.

Nuts and Bolts
</item 10>

<item 11>
FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-13T20:28:00+00:00)
Jordan Love appears to have turned a corner, meaning we should expect him to put on a show back at home. Reed will be the primary beneficiary on Sunday.

DEF New York Giants @ Saints ($3,500). There is a lot of bad offense across the NFL this season, and despite signing Derek Carr to make sure they wouldn't be
listed among them, the Saints are there anyway. The QB refuses to sit out and deal with the laundry list of injuries ailing him at this point despite them
clearly being an issue. Teammates appear fed up with Carr and would seemingly prefer the team return him to the dealership. The Saints are upside-down on this
deal right now, with no end in sight. The Giants aren't good, but they have enough defensive pop to contain an offense that keeps spinning its tires most weeks.

Best SuperDraft NFL DFS Picks for Week 15

This year, we're also listing our favorite DFS plays on SuperDraft.
</item 11>

<item 12>
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
</item 12>

<item 13>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
With an average starting position of 9.6, the 29-year-old was the sole driver to have an average starting position below 10. A win in the Round of 8 race at
Homestead-Miami propelled him to a consecutive Championship 4 appearance. Christopher Bell also dominated the qualifying races throughout the season, especially
in the playoffs. Bell recorded poles in the first three races of the playoffs to begin his playoff campaign on a superb note.

The Joe Gibbs Racing driver finished the season with two wins, 19 top-tens, and ten top-5s in 2023. He eventually finished fourth in the final NASCAR Cup Series
standings and was another one of the prominent performers of the calendar year.

4. Tyler Reddick

2023 was a year of milestones for 23XI Racing. The team had two drivers in the Cup Series playoffs for the first time since its inception.
</item 13>

<item 14>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
"I think the cars can be quieter just to help the fan experience."

Notable crew changes

Not one, but two of the biggest stars in NASCAR will have new eyes in the sky in 2024, as the spotters for both Chase Elliott and Ryan Blaney announced they
will be stepping down. Eddie D'Hondt announced Saturday he would not return as Elliott's spotter next season, followed Monday by Josh Williams announcing he
would step down as Blaney's spotter.

According to Bob Pockrass of Fox Sports, former NASCAR Xfinity Series driver Tim Fedewa, the longtime spotter for Kevin Harvick, will take over as the spotter
for Blaney next season. A possible landing spot for Williams exists at Spire Motorsports, as Williams has previously spotted for Zane Smith in the Craftsman
Truck Series.

Spire Motorsports announced a number of personnel moves this week, hiring Stephen Doran as the crew chief for Smith's No.
</item 14>

<item 15>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
Autosport has revealed its annual list of the 2023’s Top 50 drivers. The renowned motorsports publishing brand has continued the tradition since 2002, and every
year, the announcement creates some suspense within the racing community. Highlighting the 50 standout performers in motorsports for the year, Autosport has
included several NASCAR drivers in the prestigious list.

America’s Favorite Video Today

William Byron leads the list of NASCAR drivers with a 12th-place ranking as Kyle Larson and 2023 Cup Series champion Ryan Blaney join him. The scarcity of
NASCAR drivers in the rankings has raised some eyebrows. With the year producing numerous cases of NASCAR drivers standing out with their consistency and skills
behind the steering wheel, let us look at some of the major exclusions from Autosport’s Top 50 Drivers of 2023.

1. Chris Buescher

ADVERTISEMENT Article continues below this ad

RFK Racing found a fresh breath of motivation this year.
</item 15>

<item 16>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Jets would have taken possession with barely more than a minute remaining, no timeouts and Zach Wilson behind center facing a ferocious Eagles pass rush.
Was having Hurts pass in a situation when gaining a first down is unlikely the smart play under the circumstances?

“Are coaches not willing to go against the slight statistical benefit that WPA (win probability added) provides?” the exec asked. “Is it no longer OK to let
your defense win the game when it clearly has a huge talent/performance advantage? This Eagles game was 14-12, not 37-35.”

Critics attuned to the trend toward aggressiveness pounce when teams pass up chances to go for the kill. Coaches know team owners tend to be fans, not football
strategists. They know owners listen to these criticisms. That can provide an incentive to make the aggressive play, to go down swinging.

“Everybody thinks they are playing Brady, Manning or Rodgers in two-minute,” a veteran coach said.
</item 16>

<item 17>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
USA Today via Reuters Jul 30, 2023; Richmond, Virginia, USA; NASCAR Cup Series driver Chris Buescher (17) celebrates in Victory Lane after winning the Cook Out
400 at Richmond Raceway. Mandatory Credit: Amber Searls-USA TODAY Sports

Overall, he has been one of the season’s standout performers and displayed incredible grit and consistency throughout the year. His performances in 2023 make a
strong case for a place in the best 50 drivers of the year.

2. Denny Hamlin

Finding a fan of Denny Hamlin in the NASCAR fandom is a tough task. Continuously surrounded by controversy, Joe Gibbs Racing‘s #11 Toyota Camry driver has been
subject to a lot of criticism. However, his ability as a race car driver has never been in doubt. As the season entered its decisive phase of the playoffs, it
seemed like 2023 would be the year Hamlin’s championship drought finally ended.
</item 17>

<item 18>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 18>

<item 19>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 19>

<item 20>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 20>

<item 21>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
GO DEEPER Behind the AI magic that lets Amazon's Prime Vision show the NFL like never before

• Buffalo Bills: Are the defensive injuries too much to overcome?

What I think: The rest of the conference is the key variable. The Bills are diminished. They have imperfections. But Buffalo already defeated the Dolphins. The
other AFC teams aren’t exactly peaking. Kansas City and Cincinnati appear easier to defend this season than in the recent past. We’ll see how well the Dolphins
hold up. This can still be Buffalo’s season.

Exec comment: “Matt Milano and Tre’Davious White, those are huge injuries for that football team. The Bills are not as dominant as they felt going into it with
so many injuries on defense. But the division they are in, they will still be able to slug it out and get to where they need to go. Nobody has really emerged in
the AFC to unseat these teams at the top.”
</item 21>

<item 22>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
But they also haven’t really beaten anyone. They were down 14 points to Carolina. Do you think they are coming back from 13 down against Philly? In my opinion,
no.”

Advertisement

• Dallas Cowboys: Is this team any better than previous Dallas teams that fell short?

What I think: It’s a tough time to evaluate the Cowboys fairly because they were so bad against San Francisco on a big stage in their most recent game. I just
don’t see where Dallas is better or going to get considerably better, especially on offense. And every time they enjoy success, they seem to enjoy it a little
too much. That is a reflection of their owner.

Exec comment: “Run defense and consistency on defense, and then offensively, what is their identity? They just are not very good. Their best games are when they
play great defense and get turnovers, but that is not going to happen enough, especially against the good teams.”

• Detroit Lions: Have they fixed their defense?
</item 22>

<item 23>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
Is the new media rights deal a game-changer for NASCAR?

ADVERTISEMENT Article continues below this ad

Just a couple of days back, NASCAR finalized a $7.7 billion deal with Fox Sports, NBC, Amazon, and Warner Bros. Discovery. With such a lucrative deal securing
seven years of revenue, surely it seems like a lot of money. But is that really the case?

NASCAR secured almost a 40% increase over their current media rights deal yet there’s something that doesn’t fit perfectly. Sports Business Journal journalist
Adam Stern recently cleared a few things out when he attended the Dirty Mo Live podcast. With the Cup Series already facing a 5% drop in viewership this year,
Stern discussed how the new deal could impact the ratings.

At first, the renowned journalist explained how the sanctioning body would have loved to stay with Fox Sports and NBC. However, given that it would have most
likely resulted in less money, NASCAR opted to explore other options.
</item 23>

<item 24>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Detroit Lions: Have they fixed their defense?

What I think: The Lions are 11.3 EPA per game better on defense from last season to this season, the second-largest gain for any team since last season. The
schedule has something to do with that, but the 2022 defense was so bad, it couldn’t stop Carolina. Detroit will win the NFC North and return to the playoffs,
which will make this season successful no matter what happens from there. But we won’t find out until the postseason just how much better the defense might be,
because the remaining schedule features so many weak offenses. That includes Chicago twice, Green Bay, Denver and Las Vegas.

Exec comment: “Yes, I do buy them. This is a league full of teams who do not really know who they are week to week. Detroit knows who they are, and they play to
their identity. Anytime you know who you are and play to that style, you have a chance to be dangerous.”
</item 24>

<item 25>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
The winger came firing out of the gate, but the outlook has changed for the Devils in recent days after losing the superstar center Hughes to injury. Hughes
went down after slamming into the boards against the Blues, forcing New Jersey to change things on the fly. The changes have directly impacted Bratt, who was
featured on the top line alongside Hughes. With fellow center Hischier also out with an injury, leaving the Devils fairly weak down the middle and trusting
Michael McLeod to hold down the fort in between Bratt and Tyler Toffoli. As a result, Bratt has been held without an assist in each of the last two games for a
team that's missing a key contributor on offense. And it'll need him to look to score rather than play the role of playmaker. Until the Devils figure things
out, we can't trust this team to do anything positive in the offensive zone.

Win BIG with SuperDraft!
</item 25>

<item 26>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
As for this season, not so much.

Sean Payton should be at the forefront of the conversation. It's one thing to turn around a team that was dreadful last season. But he has taken one that got
off to a horrible start and turned things around in-season.

Oh — and he has resurrected Russell Wilson's career.

That alone should put him at the top of the list, but he may need to get Denver into the playoffs to convince voters.

Comeback Player of the Year

Current favorite: Demar Hamlin

Longshot: Russell Wilson

Coming back from a life-threatening injury like a cardiac arrest makes Hamlin a great candidate to win CPOY. However, it would be understandable if voters were
to look elsewhere since Hamlin has seldom been active on game day and rarely seen the field.

It will depend on how vital voters deem playing to be.

Wilson was supposed to be the next great franchise quarterback for the Broncos when they acquired him prior to last season.
</item 26>

<item 27>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
Two more drivers to have featured in the 2023 NASCAR season and Autosport’s Top 50 Drivers are Supercars racers Shane van Gisbergen and Brodie Kostecki. While
the pair battled for the championship in the Australian racing circuit, they also raced at the Indianapolis road course race earlier this season. SVG had
earlier created history by winning in his NASCAR debut at the inaugural Chicago Street Race.

ADVERTISEMENT Article continues below this ad

These are some major exclusions that could have been featured in Autosport’s annual list. At the same time, motorsports is a vast field with elite competitions
taking place the world over. Some fan favorites are bound to be left out.
</item 27>

<item 28>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
They are gaining more than 15 yards on just 11 percent of pass plays, down from 21 percent through six games last season and 19 percent for 2022 overall. Jalen
Hurts, A.J. Brown, DeVonta Smith and Dallas Goedert are still on the team, so I’m inclined to trust the production will return. But there’s also a lingering
fear defenses have caught up to aspects of the Eagles’ scheme. Does Philly have a counterpunch?

Advertisement

Exec comment: “People are going to crush Hurts and the offense. They are just churning the clock so much. It just feels like they are imposing their will on
teams through the run game and through their short passing and it just hasn’t taken off from an explosive passing standpoint, so when they play a team that can
move the ball on them through the air, it presents some challenges.”
</item 28>

<item 29>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
Since Brad Keselowski arrived in 2022, the team has continued evolving on and off the track. While Keselowski has certainly instilled his experience and winning
mentality within the team, he failed to produce a race win in 2023. Instead, it was Chris Buescher who outshone the Cup Series veteran with his best-ever season
in NASCAR’s premier competition.

Chris Buescher ended his season with three wins, nine top-5s, and 17 top-tens. Two of his three wins in 2023 came in succession at Richmond and Michigan.
Buescher continued his dream run in the Cup Series playoffs and eventually finished 7th in the final standings. The #17 Ford Mustang driver could not, however,
secure a win in the Round of 8 and did not have enough points to make a case for the championship four race at Phoenix.
</item 29>

<item 30>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
He said, “NASCAR is in a tough situation because they had to deliver more money as well and clearly if they went this route, that’s a good indication that’s
because they had to go this route to deliver more money for the industry.”

While NASCAR secured an impressive deal, there could be downsides to it and Stern explained it beautifully.

Adam Stern points out a key potential drawback with NASCAR’s new media rights deal

Safe to say, NASCAR has been struggling to keep their teams happy in terms of profitability. Since the introduction of the Next Gen cars, it has only become
more difficult for the Cup Series teams and teams in other competitions. While they have taken a commercial hit, they have piled up pressure on NASCAR to help
out in this case.

ADVERTISEMENT Article continues below this ad

In addition to that, Sports Business Journalist Adam Stern detailed one important issue with the new media rights deal. Speaking on the same podcast, he said,
“When you look at the deal with Warner Bros.
</item 30>

<item 31>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 31>

<item 32>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
Stroud in the league's concussion protocol and Nico Collins (calf) joining Tank Dell (fibula) on the Texans' injury list, Houston suddenly finds itself without
many impact skill-position players. Singletary, however, has been pretty damn good and showed us last week that he's the main man over Dameon Pierce in DeMeco
Ryans' backfield. He had 13 carries for 65 yards and a TD against the Jets, while Pierce totaled just five touches for seven yards. Don't be scared of
Tennessee's run D because it ain't what it used to be under Mike Vrabel. Raheem Mostert and De'Von Achane just posted 35.1 PPR points against the Titans, Chuba
Hubbard had 20.2 a few weeks ago, and Rachaad White had 17.8 against them in Week 10.
</item 32>

<item 33>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
I'm pretty certain they're not bad enough to make Tim Boyle look like Tim Tebow or even Tim Hasselbeck, but I do wonder if they're juuust bad enough to give
better-than-normal numbers to Garrett Wilson and Breece Hall. Reinforcements along the O-line should help the Jets' cause against a Falcons defense that's
blitzed at a league-average rate, is eighth-lowest in pass rush pressure rate and is fifth-lowest in sacks (28).
</item 33>

<item 34>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Eagles come to mind after falling 20-14 to the Jets.

Most would agree teams have gotten smarter in their handling of fourth-down situations, going for it when it makes sense, instead of punting almost out of
habit. Making the aggressive play to win can absolutely be the smart play, regardless of results. It’s progress when teams seriously wrestle with these
questions, not just on fourth down, but in all aspects of their strategy.

But when the Eagles tossed a killer interception on third-and-9 from their own 46-yard line while protecting a 14-12 lead against the Jets with two minutes left
in the fourth quarter, a longtime NFL exec reached out with a question: “Have we come too far with coaches wanting to ‘win’ the game instead of sometimes being
more traditional?”

His point: Philadelphia could have run the ball on third-and-9, then punted from around midfield.
</item 34>

<item 35>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
It wouldn't make much sense for a four-win team to rush back its franchise player.

Patriots dual-purpose back Rhamondre Stevenson (ankle) suffered an injury on a nasty hip-drop tackle in New England's Week 13 loss to the Chargers. Bill
Belichick and company already noted that 'Mondre would miss multiple weeks, so we're bumping Ezekiel Elliott way up against the Steelers.

D'Andre Swift took a hard hit and exited Philadelphia's game late against the 49ers, but it's tough to know if he's actually hurt or was just kept out because
of the lopsided score. If Swift can't go in Week 14 against Dallas, Kenneth Gainwell and Boston Scott will split touches, with Gainwell likely garnering flex
value.

Jonathan Taylor underwent thumb surgery last week and is expected to miss a few weeks. Tough for Taylor owners to ride the home stretch of the regular season
without JT, but at least they could get him back for the end of the fantasy playoffs.
</item 35>

<item 36>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
With Zach Wilson reportedly reluctant to play and further destroy a career hanging on by a thread, that leaves us with Tim Boyle, Trevor Siemian, or the lucky
fan from Section 139 to lead the Jets to yet another embarrassing defeat. As the Texans try to make a playoff push, this one could get ugly.

Best SuperDraft NFL DFS Picks for Week 14

This season, we're also listing our favorite Week 14 DFS lineup on SuperDraft, with some of our picks and analysis overlapping.

For those unfamiliar, SuperDraft's DFS scoring involves point multipliers as opposed to traditional salary-capped or tiered contests. A winning lineup will
likely feature a strong mix of stud fantasy contributors with 1x-1.6x boosts, as well as a few matchup-based sleepers with larger multipliers. Each player is
assigned a scoring multiplier based on their value as determined by SuperDraft. Users can roster any player in their lineup with no restrictions.

Win BIG with SuperDraft!
</item 36>

<item 37>
Yankees Gear Up for 2024 Season After Horrendous Year, But Could Letting This Star Go Spell Another Big Mistake? (Essentially Sports, 2023-10-10T22:30:19+00:00)
The baseball season for the New York teams didn’t turn out to be their most successful one. Both the New York Mets and the New York Yankees failed to make it to
the playoffs and came under scrutiny as the highest spenders in the league. But the level of accounting has come higher for the 27-time winners because of the
highest expectations. The management has comprehended the situation and is actively seeking to make changes for the 2024 season. Be it trimming the squad and
releasing a few players, including superstar Gleyber Torres, they look to bring about changes. But given that they are already short on options, it brings up
the question: Is the move justified?

America’s Favorite Video Today

Although it was a decent season for the second baseman, the lack of consistency brought Torres under scrutiny for changes for the upcoming season. Also, given
he is currently in arbitration with the Yankees, it adds to his trade chances.
</item 37>

<item 38>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
He also led 470 laps throughout the season and recorded his best-ever finish in the Cup Series. Reddick was among the dark horses for the 2023 NASCAR Cup Series
title and certainly held the potential to win the competition.

5. Ross Chastain

Ross Chastain‘s performance in 2023 was one of the many positives for Trackhouse Racing this season. The 30-year-old Floridian created history in the season
finale by becoming the first non-championship driver to win the Championship 4 race since the introduction of the new playoff system in 2014. Driving the #1
Chevrolet Camaro, the “Melon Man” equaled his career-best tally of two wins in a season after victories in Nashville and Phoenix.

USA Today via Reuters Nov 5, 2023; Avondale, Arizona, USA; NASCAR Cup Series driver Ross Chastain (1) celebrates his race victory following the Cup Series
Championship race at Phoenix Raceway.
</item 38>

<item 39>
FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:17:00+00:00)
DEF New Orleans Saints vs. Panthers ($4,300). The Panthers are and have been a complete disaster this year, only shielded by the fact that they play in
Charlotte rather than the Boston and New York areas. They have a QB who can't compete with the extreme lack of talent up and down the roster, a lame-duck
coaching staff, and countless other factors. They're 1-11 and are playing a divisional rival that, despite being 5-7, is somehow not out of the playoff race.
New Orleans' offense is a mess, but the defense should be primed to bounce back after a rough Week 13.

Best SuperDraft NFL DFS Picks for Week 14

This year, we're also listing our favorite DFS plays on SuperDraft.

For those unfamiliar, SuperDraft's DFS scoring involves point multipliers as opposed to traditional salary-capped or tiered contests.
</item 39>

<item 40>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.
</item 40>

<item 41>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
If he's out, Boyd is a must-start, and you can try one of Cincinnati's secondary receivers in deep leagues.

WEEK 10 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is T.J. Hockenson playing Week 10?

UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.

UPDATE 2: Hockenson is officially ACTIVE.

Hockenson (ribs) is listed as "questionable" for Week 10, but he's already said he plans to play on Sunday. The tight end has dealt with his fair share of
injury woes this season, however, he's yet to miss a game.
</item 41>

<item 42>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
He is regulated to pass from the pocket, but you are taking away the biggest weapon the offense had when you do that. Odell (Beckham) is not that guy anymore.
Who are you truly afraid of on that offense? Nobody but Lamar, and if you are not using Lamar in a certain capacity, he becomes just like the rest of these
quarterbacks around the league — hit or miss.”

• Jacksonville Jaguars: Are they ready for prime time?

What I think: Trevor Lawrence’s knee injury was the big concern coming out of Sunday. If he’s out, expectations shift dramatically. More than the other AFC
contenders, the Jaguars seem early in their life cycle, still finding their way. That makes them a little harder to trust in big spots but also leaves greater
space for growth. Let’s hope those test results on Lawrence’s knee come back clear.

Exec comment: “I like Jacksonville. Doug (Pederson) has a championship pedigree. This team is relatively young. They are scrappy.
</item 42>

<item 43>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They also seemed invigorated for Jonathan Gannon on defense and special teams. Look for the Cardinals to finish strong and play a whole lot of spoiler.

30. New England Patriots 2-8 (28)

The Patriots tried to win a game by hiding Mac Jones and their quarterbacks in general. They almost pulled it off with the running game dominating and
short-handed defense making a lot of plays, but they didn't see the results in Germany ahead of a tough bye for Bill Belichick.

31. New York Giants 2-8 (30)

The Giants can just flush away this season without Daniel Jones and Tyrod Taylor. The injuries go to their offense early and remained relentless, and now their
defense is falling apart, too. The question is, will Brian Daboll get a massive mulligan after his rookie playoff leadership?

32.
</item 43>

<item 44>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
We’ve seen the Buffalo Bills incur devastating injuries. Here we’ll address 10 teams with the shortest Super Bowl odds, running through what I think and what
NFL insiders have to say.

The full Pick Six menu this week:

• Unanswered questions for contenders

• When “playing to win” goes too far

• Championship roster, rookie kicker

• Do they make wristbands for coaches?

• Giants, Vikings and winning too soon

• Two-minute drill: Wackiest win in a while

GO DEEPER The Eagles' 'tush push' play is taking over the NFL. But is it working?

1. Where do the top 10 current Super Bowl contenders stand one-third of the way through the season? Let’s hit key questions facing the Chiefs, Eagles, 49ers,
Bills, Bengals, Cowboys, Ravens, Lions, Dolphins and Jaguars.

You can scroll through the latest Super Bowl future odds from BetMGM below.

• San Francisco 49ers: What do they have in Purdy?
</item 44>

<item 45>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Randy Mueller raised the thought last week, recalling examples from his career as a GM when offering a pick one round higher than a prospect’s actual draft
round proved alluring. Detroit used a third-round choice for Hooker, who is recovering from ACL surgery. Hooker would have been valued higher if healthy. Could
that be true again at some point in the future?

• News that Indianapolis Colts rookie quarterback Anthony Richardson might undergo season-ending shoulder surgery could carry a silver lining. Richardson got to
play enough for the Colts to evaluate where he stands, but if he shuts it down for the season, neither he nor the team would have to suffer through a full
season of growing pains. Gardner Minshew’s three-pick game against Jacksonville on Sunday works against the idea Indy is better positioned to compete with the
journeyman in the lineup, but most coaches and execs I’ve spoken with believe that to be the case in the short term.
</item 45>

<item 46>
Kyle Larson Breaks Down His Sprint Car Series Journey and How He Partnered Up With Dale Earnhardt Jr.’s Former Pupil (Essentially Sports,
2023-09-28T21:31:48+00:00)
Events like these also help to keep these smaller race tracks across the country in good shape. If a track isn’t used or taken care of for an extended period of
time, it will start to wear out and eventually become completely unsuitable for any kind of racing unless it is taken apart and rebuilt. But one of Kyle
Larson’s reasons for doing what he does with the High Limits Racing Sprint Car Series is a lot simpler, and he revealed that goal to Dale Earnhardt Jr. at the
Lernerville Speedway.

“Can’t Have Everything You Want”- Dale Earnhardt Jr. Ruthlessly Trashes Home Track as He Endorses Wild NASCAR Rumors

“I was hoping this would add pressure to the weekend shows to raise their purses as well so I think we’ve seen a big step in that this year and I think it’s
gonna continue forward so, I would say so far it’s been successful,” he added.
</item 46>

<item 47>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
3. Christopher Bell

As unfortunate as Denny Hamlin was, his JGR teammate Christopher Bell must have felt hard done by after an impressive season. The #20 Toyota Camry driver
progressed to the championship race for a second successive season. However, a cruel twist of fate waited for him in the season finale at Phoenix Raceway, as he
was forced to retire early from the biggest race of his career with a broken brake rotor.

via Imago NASCAR, Motorsport, USA Food City Dirt Race Apr 9, 2023 Bristol, Tennessee, USA NASCAR Cup Series driver Christopher Bell 20 wins the Food City Dirt
Race at the Bristol Motor Speedway Dirt Course. Bristol Bristol Motor Speedway Dirt Course Tennessee USA, EDITORIAL USE ONLY PUBLICATIONxINxGERxSUIxAUTxONLY
Copyright: xRandyxSartinx 20230409_lbm_bs1_024

Bell’s 2023 season has gone under the radar.
</item 47>

<item 48>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
The 43-year-old dominated the first round of the playoffs and was the driver to watch out for. He eventually won his first playoff race in Bristol and cruised
to the next round. However, Hamlin revealed recently that a shoulder injury prevented him from being at the best of his abilities as the playoffs progressed.
Having to depend much on his weaker arm, Hamlin stayed in contention for the championship four race until the final race in Martinsville.

He secured three wins, 14 top-fives, and 19 top-10 finishes in 2023. He also led a whopping 998 laps, which makes his omission from Autosport’s list seem
unjustified. He finished 5th in the final standings and was one of the most consistent and best performers of the 2023 NASCAR season. From the looks of it, a
championship trophy would have surely put him in the upper echelons of Autosport’s Top 50.

3.
</item 48>

<item 49>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
This team is relatively young. They are scrappy. The fact that they have been there before gives them a chance. I think the quarterback is growing.”

• Cincinnati Bengals: Can Joe Burrow hold up for a full season?

What I think: Burrow is looking healthier over the past two weeks. The bye week comes at a great time for Burrow to get even healthier, but check out the
schedule on the other side: San Francisco, Buffalo, Houston, Baltimore, Pittsburgh and Jacksonville. Cincy still has a second game with the Steelers, plus games
against Kansas City and Cleveland. I’m worried Burrow will wear down against these defenses.

Exec comment: “You cannot always expect Joe Burrow to create magic with a bottom-half offensive line. The line is my concern all the way.”

Advertisement

2. Are teams sometimes going too far playing to win? The Eagles come to mind after falling 20-14 to the Jets.
</item 49>

<item 50>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
The team had two drivers in the Cup Series playoffs for the first time since its inception. One of the major reasons behind its success in 2023 is #45 Toyota
Camry driver Tyler Reddick.

Alongside team owner Denny Hamlin, Tyler Reddick was the pick of the drivers in the first round of the NASCAR playoffs. The 27-year-old California native
showcased his road course prowess with a win at COTA earlier this year. Carrying the momentum to the playoffs, Reddick won in Kansas in the Round of 16 to gain
more traction for his championship push. He also had one of the best average starting positions among the drivers, with an average of 10.167, despite having
only two poles to his name.

Watch This Story: NASCAR legend Kevin Harvick’s unexpected EV confession

Tyler Reddick finished sixth in the final standings and ended the season with two wins, ten top-5s, and 16 top-10s.
</item 50>
=== END ITEM id=ccfcbf48ccf4fa20 ===

=== ITEM id=8671118aefbc9ac8 ===
Gold sentences and facts:
Sentence 0: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 0.0: Epic chose to focus on simply painting Google as the bad guy on day one.
  Fact 0.1: Epic's focus on painting Google as the bad guy happened on day one of the trial.
  Fact 0.2: The issue of painting Google as the bad guy will probably come up later in the trial.
Sentence 1: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 1.0: Epic will attempt to argue that Google restrains competition within two separate markets.
  Fact 1.1: One of the two markets is the distribution of apps to Android users.
  Fact 1.2: The other market is the market for payment processing solutions for content inside Android apps.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?
</item 1>

<item 2>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.
</item 2>

<item 3>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 3>

<item 4>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 4>

<item 5>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Well… both sides lost! But Epic arguably lost more. Even though Apple has incredible power over the iPhone, Judge Yvonne Gonzalez Rogers ruled the company
didn’t have an unfair monopoly in this case – partly because she decided the relevant market for Fortnite was “digital mobile gaming transactions” rather than,
say, iPhone apps. She also decided that Epic violated its developer agreement with Apple and would have to pay.

But she also barred Apple from keeping iPhone users in the dark about alternate ways they can pay for apps – and may have even allowed developers to add their
own payment mechanisms. I won’t dwell on the Apple ruling, as I’m ethically bound, but my colleague Adi Robertson has a comprehensive breakdown.

How can Epic possibly have a case against Google when Apple already won?

Epic declined to answer this on the record, among other questions we asked... but three things to consider:

1) That was a different case. Also, that one’s not over till the Supreme Court weighs in or declines to take a look.
</item 5>

<item 6>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
The competitive landscape with other app stores, OEMs

The tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point out that not only does
the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also supports competitive app
stores like Samsung’s Galaxy Store and Amazon’s Appstore.

Epic, however, will attempt to demonstrate that Google makes it hard for alternative app stores to succeed and reach consumers, noting that 90% of apps are
still downloaded through Google Play. And it will point out that Google bundles the Play Store with other Google apps that Android OEMs (original equipment
manufacturers) have to agree to in order to use Android.

Epic will also bring up “Anti-Fragmentation Agreements” (AFAs) that prevent the OEMs from modifying Android to allow for frictionless downloading of apps
outside of Google Play.
</item 6>

<item 7>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
No, we’ve always turned down special deals just for Epic. We’ve always fought on the principal that all developers should be, you know, given the same
opportunities. One of the documents in evidence was a 2019 email I sent — right before Fortnite launched the Marvel season with all this amazing new content and
the Star Wars event featuring J.J. Abrams. Right before that, I sent an email to all the Google senior executives saying that we wanted to bring Fortnite to the
Google Play Store in time for that event, and we wanted them to allow us and all other developers to process certain payments and keep them all.

That was our proposal to Google in 2019. If Google had said yes to that, that would have been awesome for all developers — the Android ecosystem would have
become much, much stronger, and Google would be in a much better position in the smartphone industry than they are today. We would’ve never had a dispute
because the problem would have been solved.

It’s always been in Google’s power to solve this problem.
</item 7>

<item 8>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
[Editor’s note: Every journalist in the courtroom agreed the jury was attentive; most were also surprised by the rapid verdict.]

They got it, they got it quickly, and they were able to pull apart what was going on and contrast it with the fiction of the story that Google was trying to
tell.

What was something that surprised you in the courtroom, something you heard come to light that you hadn’t heard before?

I’d suspected a lot of the practices that Google had, you know, since 2018 or so when we first started this, to such an extent that some folks would
occasionally call me a conspiracy theorist. It was really, really interesting to see that my understandings of what Google was doing behind the scenes were
actually true — you’re leaking our conversations to reporters to get negative stories written about us; you’re paying other developers off to convince them not
to launch their own stores; they were going around and paying carriers and OEMs secretly not to carry competing stores.
</item 8>

<item 9>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What were your settlement talks with Google CEO Sundar Pichai like?

We can’t talk about content, but we met for an hour and had a professional discussion... in which we didn’t reach a settlement. We were rather far apart, let’s
say, because what Epic wants ultimately is free competition and fair competition for everybody, and the removal of the payments tie and removal of the
anticompetitive measures, which obviously leads to far better deals for consumers and developers.

What we don’t want is a special deal just for ourselves, and Google’s strategy so far as you’ve seen from all these deals has been to basically take out all
their potential competitors one at a time while they’re still weak and small, before they build up or unite into a formidable force. Epic’s not going to go
along with something like that and do a special deal just for ourselves.

You’ve said that a couple times, but the email that you accused Google of leaking suggested that Epic was interested in a special deal of some sort at that
time.
</item 9>

<item 10>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 10>

<item 11>
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
</item 11>

<item 12>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
The lawyers can see them, but I saw almost all the key elements in this case at the same time the jury saw them. Two things stand out big time. Google’s Project
Hug was an astonishingly corrupt effort at a massive scale — the same corrupt deal structure they tried to do with Epic when they wanted to pay us off to launch
Fortnite on Google Play and not launch our own store.

As soon as we thwarted their effort, they went around to 27 different developers and offered each one a payoff to undermine any effort we had to get their games
onto our store exclusively. Activision and Riot and Supercell had direct distribution plans that they were planning on; Google paid them not to pursue those
plans. Just direct blatant violations of anti-competition law, it’s crazy a company of Google’s scale would do that.

“If you’re a smaller developer than Spotify, you get screwed.”

The other astonishing one was the Spotify deal. Spotify is the one company that had comparable negotiating power to Fortnite.
</item 12>

<item 13>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
As an example, Epic will point to Activision Blizzard, the owner of King.com and maker of Candy Crush and Call of Duty, as one developer that wanted to go the
route of opening its own app store. Google and Activision Blizzard then entered into a three-year agreement in January 2020, worth approximately $360 million,
to keep the developer on Google Play, Epic will argue.

Google, however, will characterize this program as a way to incentivize developers to launch their apps on Android at the same time as they launch on iOS and
plans to dispute Epic’s claims that it prevented developers from opening their own app stores.

Android allows sideloading

Google will make note that, unlike Apple, it also allows apps to be sideloaded on Android devices — something Epic takes advantage of today. It will note this
process is streamlined and helps keep users safe.
</item 13>

<item 14>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Also, that one’s not over till the Supreme Court weighs in or declines to take a look.

2) Google can’t tell jurors that Apple won its case, or that other plaintiffs settled. The judge in this case says so!

3) Oh, and by the way, this is a jury trial.

Wait, why does that matter?

Epic and Google have to convince a jury, not the judge, which is totally different from how the Apple case played out. (That one was a “bench trial.”)

Maybe all the evidence of tricky deals inside Google might sway a jury against the company? Maybe Google scaremongering that sideloaded apps equal gaping
security holes will sway a jury against Epic instead? Who knows!

(If you’re a Epic juror reading this — stop! Judge Donato explicitly said you’re in a “news-free bubble” through mid-December, folks.)

Didn’t other parties sue Google too?

They did! And then, they all settled.
</item 14>

<item 15>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 15>

<item 16>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 16>

<item 17>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What would you say the differences are between the Apple and Google cases?

I would say Apple was ice and Google was fire.

The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the
same terms, they force OEMs and carriers to all have the same terms.

Whereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re
paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and
it’s right there for everybody to read and to see plainly.

I think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing,
whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute.
</item 17>

<item 18>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 18>

<item 19>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Or at least it was worse, before they were destroyed.”

The best Google could do in response was to plant its own feeble seed with the jury, too: “Is Epic using the chats to distract me from all the evidence I do
see?”

“It’s true that Google could have automatically saved all chats for all relevant employees, but just because Google didn’t save some chats didn’t mean it
violated antitrust laws,” Pomerantz argued.

Epic’s opening statements seemed to paint a clearer picture for the jury than those from Google. But things got complicated for both parties when the first two
witnesses — Epic Games Store head Steve Allison and Yoga Buddhi CEO Benjamin Simon, who also appeared in the earlier Epic v. Apple trial — took the stand.

Both Epic and Google spent a long, long time on subtle lines of questioning.
</item 19>

<item 20>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
First, Epic announced it was bypassing Apple and Google’s app store fees. It deployed a hotfix update to Fortnite without either company’s knowledge, letting
you purchase V-Bucks directly through its own payment processing option at a discount. Apple and Google almost immediately reacted by kicking Fortnite off their
app stores for breaking the rules.

Then: surprise! Epic was ready and waiting with two lawsuits and an attack ad, depicting a Fortnite hero throwing a unicorn-llama hammer into a giant screen
reminiscent of Apple’s famous “1984” Macintosh ad.

It was a striking publicity blitz followed by a lot of slow-moving court proceedings. While the Apple lawsuit went to court in 2021, the Google one was delayed
again and again. A ruling came down for the Apple trial that September, and it was mostly decided in Apple’s favor, though both parties are waiting for the
Supreme Court to potentially weigh in. Meanwhile, the machinations for its fight against Google continued, and now…

It is time for trial number two.
</item 20>

<item 21>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Thou shalt not have a competing store on iOS and thou shalt not use a competing payment method. And I think Apple should be receiving at least as harsh
antitrust scrutiny as Google.

It’s interesting to me that because Google distributes the Android operating system as open source, they had to put all these deals out in the open. More out in
the open, I should say — certainly they still wanted to keep them secret.

But I’m going down my story about all the best emails from the Epic v. Apple trial — and we do have a lot of documents from both Apple and Google that show they
were similarly self-serving in terms of deals.

I’d say this is the thing that’s disappointed me the most with Apple and Google: even at the peak of the antitrust trial against Microsoft, Microsoft was
awesome to developers. Microsoft has always been awesome to developers, always being respectful, giving developers a great deal and treating them as partners,
you know? And so even as Microsoft was crushing corporate competitors, the developer experience was excellent.
</item 21>

<item 22>
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
</item 22>

<item 23>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
And when we tried to bundle Fortnite with other smartphone manufacturers like OnePlus and carriers of all sorts, they told us they couldn’t do a deal because
Google had done a secret deal with them.

It was really disconcerting to see the extent of bad faith efforts that were going on in a company of Google’s size. You’d think a trillion-dollar company would
develop to the point where they have pretty respectable processes and leadership structures that provide a check and balance against wrongdoing, but they were
rampantly destroying all their chats on these topics.

You’d see long conversation threads would start to get into a spicy antitrust issue, and suddenly somebody points out the history’s on and the chat goes silent.
They just turned it off to have the documents destroyed. It was great to see that all called out in detail.

Any particular evidence where you’re like, “Oh wow, I never saw that, and now that specific thing came out for the world to see”?

As an employee of Epic, I’ve not been able to see Google’s internal documents until the trial started.
</item 23>

<item 24>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.
</item 24>

<item 25>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
I don’t know if that was on behalf of you and fellow developers. Can you tell us what that email was actually about?

Google didn’t leak the email; they leaked their ridiculously biased summary of it to Abner Li at 9to5Google. Don Harrison testified at trial that he didn’t
think Google leaked to the press. And then he was presented with a document including Google’s press team and Sameer Samat, head of all Android, basically
summarizing the articles that appeared as a result of Google’s leaking of our plans to the press.

So I was wondering if you could tell me what the email actually said.

Oh, it’s in evidence! This was the one I mentioned! Right before Christmas-ish 2019, we were planning to launch the new Marvel season with a bunch of Star Wars
content and its awesome lineup. I asked Google’s executives to let Fortnite come to the Google Play Store using our own payment method and to open up, you know,
using their own payment methods to all developers, not just Epic.
</item 25>

<item 26>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic, meanwhile, lost its antitrust battle with Apple and is now asking the Supreme Court to weigh in.
</item 26>

<item 27>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.
</item 27>

<item 28>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Both Epic and Google spent a long, long time on subtle lines of questioning. You really had to read between the lines to see that Epic was trying to make a
point about how Google’s 70/30 revenue split is probably based on an arbitrary decision Valve made two decades ago with Steam or how Google was trying to make a
point that Epic, too, likely believed that an app store provides more value than just payment processing and maybe deserves more money.
</item 28>

<item 29>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Who will Epic and Google call as witnesses?

We are going straight to the top of the org chart, folks. Google CEO Sundar Pichai and Epic CEO Tim Sweeney are absolutely expected to attend. Android
co-founder Andy Rubin might be deposed, and Google says we’ll hear from current Android (and Chrome, and Chrome OS) boss Hiroshi Lockheimer. Google also tells
us representatives from Apple, Netflix, Motorola, and AT&T should testify as well.

How much hypocrisy can we expect in the courtroom?

I’d say quite a bit! For example:

Epic says it’s not in this for the money, it’s only suing to end Google’s monopoly — but Epic would surely profit if it doesn’t have to pay Google’s cut.

It looks like Epic’s going to argue that you’re locked into Android after buying an Android phone, but I can personally attest that switching back and forth
between the platforms is easier than ever.
</item 29>

<item 30>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
We know that because they came to us and told us they were doing that. And we know it because after all that went down, I talked to an Activision employee who
was close to that effort, working with the team building the store, and he reported that the entire store team Activision had employed to build their competing
store had been disbanded as soon as Activision signed their deal with Google.

[Editor’s note: We did not hear from such a person at trial; I’d be eager to speak to them now!]

We know Activision was telling Google they were building their computing store; we know in Google internal discussions they said they didn’t want Activision
building a competing store. They agreed to sign this deal, and they were gleeful about the fact they dissuaded perhaps their number one competitor at the time
from launching their own store.

Four years later, Microsoft acquired Activision Blizzard — and one of the big talking points from Microsoft to the European Union was that the merger
strengthened the company overall in order to provide a viable computing store on iOS and Android.
</item 30>
=== END ITEM id=8671118aefbc9ac8 ===
