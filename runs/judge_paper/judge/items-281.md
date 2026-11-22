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

This file has 6 items.

=== ITEM id=518f2e618ff3db99 ===
Gold sentences and facts:
Sentence 0: Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in 2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.
  Fact 0.0: A trial hearing took place on Monday.
  Fact 0.1: The Monday trial hearing revealed many tidbits.
  Fact 0.2: Google spent $26.3 billion in 2021 making itself the default search engine across platforms.
  Fact 0.3: Google tried to have Chrome preinstalled on iPhones.
  Fact 0.4: Google tried to take its default search engine position further.
  Fact 0.5: The hearing revealed other tidbits in addition to these.
Sentence 1: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 1.0: The last roundup reported that Google spent $26.3 billion in 2021 making itself the default search engine across platforms.
  Fact 1.1: The last roundup reported that Google tried to have Chrome preinstalled on iPhones.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google siphons off news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Yahoo was already well behind Google in terms of search technology at that point. But also, Firefox was very well behind Chrome, which ended 2014 with nearly
50% market share and ended 2017 (when Google got prime default search position on Firefox) at nearly 65% of all web browser usage on desktop: and mobile is even
more skewed to Chrome. In other words, the number of Firefox users might have been declining for other reasons, although pushing the blame on Yahoo certainly
works in Google’s favor here!

The trial continues…
</item 1>

<item 2>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.
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
</item 4>

<item 5>
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
</item 5>

<item 6>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Now, onto the roundup!

A window into Google’s most popular search queries

Judge Amit Mehta ruled to make a list public that provides a glimpse of which search terms make Google the most money. The list of popular search terms ordered
by revenue includes 20 terms that were lucrative for the week of September 22, 2018. Information like revenue per search term, how many queries each of those
terms got, along with a separate list of popular search terms ordered by queries (not revenue), were all redacted. The list we can see is as follows:

iPhone 8

iPhone 8 plus

Auto insurance

Car insurance

Cheap flights

Car insurance quotes

DirecTV

Online colleges

AT&T

Hulu

iPhone

Uber

Spectrum

Comcast

Xfinity

Insurance quotes

Free credit report

Cheap car insurance

AARP

LifeLock

There is, in reality, little surprise here.
</item 7>

<item 8>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Quick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be the default search engine on Firefox. At
the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.

“I felt strongly that Yahoo was not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said
Yahoo had promised to reduce the number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.

“The number of users who stayed with Firefox declined noticeably during the years when Yahoo was the default,” said Baker.

Baker, who was providing a recorded deposition for Google’s defense, noted that Mozilla’s users apparently wanted and expected Google.

However, that is not the full story. Yahoo was already well behind Google in terms of search technology at that point.
</item 9>

<item 10>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 10>
=== END ITEM id=518f2e618ff3db99 ===

=== ITEM id=17c64e632c7f0276 ===
Gold sentences and facts:
Sentence 0: 9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.
  Fact 0.0: At 9:47 p.m., the Bears answer right back and sack Dobbs.
  Fact 0.1: Sweat and Brisker are in there to take Dobbs down.
Sentence 1: The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in points (16.7) and 10th in yardage (310 per game).
  Fact 1.0: The Ravens defense has allowed the fewest points per game this season, 16.1.
  Fact 1.1: The Ravens defense has allowed the second-fewest total yards this season, 4,030.
  Fact 1.2: San Francisco's defense ranks second in points, at 16.7.
  Fact 1.3: San Francisco's defense ranks 10th in yardage, at 310 yards per game.

Retrieved items:
<item 1>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He also ducked under an end-zone target instead of leaping for it. It's worth noting that a 36-yard crosser he ran against the Chargers flies in the face of
this as he separated from man coverage with speed and technique, then turned upfield and ran outside before racing past a defender and diving for the pylon.

OSBORN: Has a 50% catch rate on the year and that includes whether he faces zone coverage or man coverage. He sees more targets per route run against man
coverage and averages more yards per catch and yards after catch against man coverage. Over his career, targets do seem to come more frequently against man
coverage and he has been more explosive with them.

BEARS: Play zone coverage at the fourth-highest rate of any defense this season (79.2%). And because Cousins is much more explosive of a passer against man
coverage, expect Chicago to keep up their zone defense.
</item 1>

<item 2>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The honeymoon period is over for Joshua Dobbs in Minnesota.

The folk hero quarterback threw four interceptions in the Vikings' 12-10 "Monday Night Football" loss to the Bears, leading a sluggish offense that didn't find
its footing until the fourth quarter in a game that Minnesota wouldn't have needed many points to win.

The Bears' defense swarmed Dobbs throughout the night, sacking him only twice but bringing a great deal of pressure as the defensive backs made plays. Despite
the barrage of turnovers, Chicago largely failed to take advantage. The Bears scored only three points off of the four Dobbs interceptions.

MORE: How Joshua Dobbs got his 'Passtronaut' nickname

Meanwhile, Chicago's first turnover of the night — a deflating fumble by Fields when the Bears had a chance to extend their lead to two possessions — was
quickly turned into a touchdown by the Vikings.
</item 2>

<item 3>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They can win with offense or defense. Dan Campbell is coach of the year.

4. San Francisco 49ers 6-3 (8)

The 49ers had a surprise offensive get-well game in Jacksonville with Brock Purdy leading the way and all his key supporting weapons doing damage, even without
Christian McCaffrey scoring. The biggest development was the defense going back to dominating a good offense.

5. Baltimore Ravens 7-3 (2)

The Ravens' offense is just fine with the running and passing of Lamar Jackson, but he still can't afford to make big mistakes, which he did vs. the Browns. The
defense needs to pick up the pieces, too, stat, ahead of facing the Bengals on a short week.

6. Miami Dolphins 6-3 (6)

The Dolphins got a much-needed bye to figure out how to beat the better teams in the league.
</item 3>

<item 4>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:00 p.m. — Will the Josh Dobbs show continue tonight? He’s had good command of the Vikings’ offense in three games since a trade out of Arizona, flashing an
especially strong connection with T.J. Hockenson.

7:45 p.m. — It’s 15 degrees in Minnesota with a feels like of 2, so the roof will come in handy tonight at U.S. Bank Stadium.

7:25 p.m. — Vikings legend Adrian Peterson is in the building to watch his former team.

7:17 p.m. — The Bears enter tonight with the No. 1 and No. 4 picks in the 2024 NFL Draft, and they'll remain in those slots with or without a loss thanks to the
strength of schedule tiebreaker.

7:10 p.m. — Heading into Week 13, tonight's game is the difference between the No. 6 and No. 7 seed for the Vikings.
</item 4>

<item 5>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in
points (16.7) and 10th in yardage (310 per game).

The 49ers defense might be slightly inferior statistically, but it could have more of an impact in this matchup. San Francisco has racked up an NFL-leading 25
turnovers and has given up the second-fewest rushing yards (1,252), which should help slow down the team with the league's most effective running game.
</item 5>

<item 6>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
— Dobbs is nearly intercepted by Jaylon Johnson (again) on third down. The last turnover wasn't Dobbs' fault, but this one should've been a pick. The Bears'
defense is all over the Vikings tonight, and Minnesota can't get out of its own way.

9:13 p.m. — Addison, trying to make up for the drop that turned into an interception, looks like he goes for 17 yards, but replay shows his knee was clearly
down. It becomes a much smaller 4-yard gain.

9:09 p.m. — Fields comes under pressure on third down and misses D.J. Moore. He's starting to miss these receivers after a hot start. Somehow, the Vikings will
have a chance to tie this game or take a lead before halftime.

9:04 p.m. — Dobbs is picked off again!
</item 6>

<item 7>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
McCaffrey, who seems poised to win the Offensive Player of the Year Award, leads the NFL in rushing (1,292 yards) and total yards per game (128.6) this season
and is tied for first in touchdowns (20).

Ravens TE Isaiah Likely thriving in the starting role: The Ravens offense hasn't lost a step since losing one of the team's top weapons in tight end Mark
Andrews (ankle) in Week 11, largely due to Likely's emergence.

The second-year tight end has averaged 64.3 yards over his last three games, which is particularly noteworthy since Andrews averaged 54.4 yards per game before
sustaining his injury.

Should we expect a defensive battle?: While both teams boast offenses that rank in the top five in yardage and scoring, this may not turn into a shootout.
</item 7>

<item 8>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Baltimore Ravens: Are the Ravens better off shifting to more of a pass-oriented offense?

What I think: Baltimore has never scored fewer points or amassed less offensive EPA through the first six games of a season with Lamar Jackson in the lineup.
The schedule and injuries have played roles, but with the Ravens shifting toward a more conventional pass offense, are the explosive quarterback runs mostly in
the past? Jackson has one rush longer than 20 yards (he’s had four or five by now previously). He’s gaining at least eight yards on 15 percent of carries, half
his previous rate. I’m not expecting a consistent precision passing game to suddenly materialize. Will this new offense be tougher to defend?

Advertisement

Exec comment: “The defense keeps them in it, but I don’t necessarily think this offensive change is going to make them any better when it counts. Lamar Jackson
does not appear as dynamic as a runner, but in this offense, he does not need to be.
</item 8>

<item 9>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
It's 4th & 10, and Chicago is lining up to got for it.

8:36 p.m. — Sacks are killing the Bears in this game. Multiple Vikings defenders wrap up Fields for a 12-yard loss and quickly set Chicago back to 2nd & 22.

8:35 p.m. — The Bears' offense is picking up where it left off. After Fields takes off for 14 yards, he hits Roschon Johnson for 22 yards to take Chicago into
Vikings territory.

8:31 p.m. — That's what the Bears got Montez Sweat to do. Sweat sacks Dobbs on third down to force a quick three-and-out and get the ball right back in Fields'
hands.

8:27 p.m. — A great start for the Bears ends with zero points as Cairo Santos misses a 48-yard field goal. That's a deflating sequence for Chicago.

8:25 p.m. — The rookie Ivan Pace Jr.
</item 9>

<item 10>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:25 p.m. — The rookie Ivan Pace Jr. finally gets to Fields and drops him for a five-yard loss. It'll be 3rd & long for the Bears.

8:24 p.m. — Fields throws on the run to find Roschon Johnson for a first down, and he quickly follows up with a 13-yard pass to Khalil Herbert. The Bears are
driving, and Fields doesn't have an incompletion yet.

8:21 p.m. — The Bears have taken it into Vikings territory with a string of short gains. For Chicago, that's progress.

8:19 p.m. — Fields finds Moore for a short gain and the first 1st down of the night.

8:16 p.m. — The Bears will start with the ball on the road, looking to build early momentum.

8:00 p.m. — Will the Josh Dobbs show continue tonight?
</item 10>
=== END ITEM id=17c64e632c7f0276 ===

=== ITEM id=e003b2cc8f8dd020 ===
Gold sentences and facts:
Sentence 0: Unfortunately, she did say that she had conversations with investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine.
  Fact 0.0: This person's statement is described as unfortunate.
  Fact 0.1: This person said they had conversations with investors as part of their due diligence.
  Fact 0.2: Alameda was taking on losses from FTX.
  Fact 0.3: Alameda was taking on losses from FTX to keep FTX's balance sheet pristine.
Sentence 1: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 1.0: Huang gave testimony during cross-examination.
  Fact 1.1: Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 1.2: Huang said Paradigm was told that Bankman-Fried did not want investors as directors.
  Fact 1.3: Huang said Bankman-Fried did plan on having a board with experts, in contrast to not wanting investors as directors.

Retrieved items:
<item 1>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”
</item 2>

<item 3>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
In late 2019, Alameda had a negative balance on FTX. Because they sat in an open-plan office, Wang heard a trader ask Bankman-Fried if it was okay to keep
withdrawing money from that account. Bankman-Fried said it was okay as long as it was lower than FTX’s total trading revenue. He was the CEO of both companies
at the time.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses.

The amount that Alameda was allowed to go negative began to creep up. Around 2019 or 2020, Wang checked the database and discovered that Alameda was negative by
about $200 million, which was more than the $150 million FTX made in revenue. That had to mean Alameda was taking customer money. It surprised him and he says
he discussed it with Bankman-Fried, who told him that he just needed to take into account the value of FTT. Wang said he trusted Bankman-Fried’s judgment and
didn’t pursue the issue further.
</item 3>

<item 4>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
That was all done by Bankman-Fried, he testified. In early 2020, Wang said he discovered for the first time Alameda's negative balance exceeded FTX's revenue,
an indication that Alameda was taking customer funds. Wang said he brought this to Bankman-Fried's attention several times. In late 2021, Wang discovered
Alameda had withdrawn $3 billion from its $65 billion line of credit. Wang's compensation was a base salary of $200,000 per year plus stock. He owned roughly
17% of FTX. Even though they were co-founders, "ultimately it was Sam's decision to make" when there were disagreements, he said.
</item 4>

<item 5>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Other customers didn’t have the same privileges.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses. In 2021, FTX was facing a “several hundred million dollar” loss from an exploit
in MobileCoin, Wang said. (This seems to have happened in April 2021, though Wang didn’t give a month.) Bankman-Fried directed him to have Alameda take it on
instead, because FTX’s balance sheets were visible to investors and Alameda’s weren’t.

Yesterday, Paradigm Capital’s Matt Huang showed us the balance sheets he received from Bankman-Fried. They didn’t include the loss, which should have been
included in the second quarter numbers he’d received — they showed $63 million in trading expenses and $63 million in “other expenses” in that quarter, and an
estimated profit of $732 million.

“FTX was not fine and assets were not fine.”

By 2022, Alameda’s negative balance on FTX was a cause for concern for Caroline Ellison, Singh, and Wang.
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Well, Bankman-Fried remembers it differently. See, Yedidia was just asking about Alameda’s risk profile, and Bankman-Fried wasn’t talking about insolvency at
all.

Also, when Singh and Bankman-Fried had the dramatic balcony conversation at their penthouse, it was just that Singh thought Alameda’s liabilities had gotten too
high, and FTX was spending too much money on marketing. But Bankman-Fried still thought that Alameda had more assets than liabilities, so it was all fine, and
besides, if Singh thought he was going to be better at marketing, he could take it over. It didn’t have anything to do with the money Alameda owed FTX at all.

Of course not! Bankman-Fried didn’t learn about the $8 billion liability associated with Alameda until October 2022, he said. And he learned it all by himself,
by looking at a computer database. When he found it, he was “very surprised!”
</item 6>

<item 7>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Those privileges ultimately resulted in Alameda owing FTX $8 billion worth of customer deposits. "We gave special privileges on FTX that gave unlimited
withdrawals on the platform to Alameda," Wang said. Alameda was allowed to withdraw and transfer those funds and had a $65 billion line of credit. "When
customers deposited USD, it went to Alameda," he said. "It existed in the computer code. Alameda could have negative balances and unlimited withdrawals." That
"bug" in the code was written by Nishad Singh, who was FTX's director of engineering, and reviewed by Wang. Bankman-Fried was calling the shots, Wang said. Wang
also told the court about a $1 million personal loan he received and a $200 million to $300 million loan in his name from Alameda that was never deposited into
his account, but rather was used to make investments into other companies on behalf of FTX.
</item 7>

<item 8>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
The one Wang said Alameda took to keep off FTX’s balance sheet? Yeah, so it was a totally innocent thing where what actually happened was that Bankman-Fried
thought it was appropriate that Alameda take the position as a backstop liquidity provider, that’s all.

In June 2022, Bankman-Fried heard about the account called “fiat@ftx” tracking how much money Alameda owed to FTX, he testified. He did not know what it was and
did not bother to find out. He was busy! That was when Bankman-Fried directed Ellison to repay Alameda’s lenders, because he thought Alameda was good for it. He
also gave BlockFi and Voyager, two crypto lenders, some capital infusions for good measure.

He was “very surprised!”

Remember that testimony Adam Yedidia gave about a conversation with Bankman-Fried in August 2022 about the enormous amount of money Alameda owed FTX? Well,
Bankman-Fried remembers it differently.
</item 8>

<item 9>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Additionally, he was promised that Alameda had no preferential treatment on the FTX platform, even though the hedge fund was one of its top traders. Huang said
he was concerned about FTX's lack of a board of directors, but he eventually invested anyway. During cross-examination, Huang said Paradigm pressed
Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts. — CNBC's Dawn Giel
contributed to this report.
</item 9>

<item 10>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.
</item 10>
=== END ITEM id=e003b2cc8f8dd020 ===

=== ITEM id=b49b08e8e25d9c41 ===
Gold sentences and facts:
Sentence 0: Last week, the Supreme Court decided that it would hear the pair of cases, which revolve around Republicans crafting state-specific laws that order platforms to keep their hands off of some social media posts.
  Fact 0.0: The Supreme Court decided last week that it would hear the pair of cases.
  Fact 0.1: The pair of cases revolve around Republicans crafting state-specific laws.
  Fact 0.2: The state-specific laws order platforms to keep their hands off of some social media posts.
Sentence 1: Under Musk, Twitter — now called X — has prioritized “free speech” managed by crowdsourced-based moderation.
  Fact 1.0: Under Musk, Twitter has prioritized free speech.
  Fact 1.1: Twitter is now called X.
  Fact 1.2: Under Musk, free speech on Twitter is managed by crowdsourced-based moderation.

Retrieved items:
<item 1>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Are the Texas and Florida laws identical?

The two laws — HB 20 in Texas and SB 7072 in Florida — are very similar in their origins and intentions, but diverge slightly in how they seek to restrict
social media platforms.

In both instances, a provision of the state law instructed social media companies to stop removing certain kinds of content. In Texas, the law told social media
companies that they could no longer remove or demonetize content based on the “viewpoint represented in the user’s expression.” In Florida, the law would stop
social media companies from banning political candidates or removing or restricting their content. The laws have a few other provisions, but the idea is that
conservative politicians in those states want to regulate how tech companies interact with political content.

“They’re in the same ballpark, the sentiment is the same,” Barrett said.
</item 1>

<item 2>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
“They’re in the same ballpark, the sentiment is the same,” Barrett said. “Republican lawmakers in each state believe that — and said explicitly in the course of
debating and passing these laws — that ‘Silicon Valley oligarchs’… are ideological liberals, and they are censoring people in our states who are conservative,
and we are hereby ordering them to stop doing that.”

How could the Supreme Court decision affect social media companies?

If the court finds that social media companies don’t have a First Amendment right to curate the kinds of content that they allow, social platforms could look
very different, at least in states that are trying to limit their moderation powers. After years of slow progress on misinformation — and worrisome backsliding
on platforms like Elon Musk’s X — the Supreme Court’s decision could upend that process, sowing chaos online in the process.
</item 2>

<item 3>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
What does this have to do with the First Amendment?

The case revolves around First Amendment rights — but, counterintuitively, it’s the rights of social media companies that are in question, not the rights of
their users.

“The question here is: Do social media companies have a First Amendment protected right to exercise what you might call editorial judgment?”

“These cases are about the First Amendment and how the First Amendment applies to social media companies. And then more specifically, what the First Amendment
has to say about content moderation, which is obviously a subset, although a really big and important subset of what social media companies do,” Barrett said.

“The question here is: Do social media companies have a First Amendment protected right to exercise what you might call editorial judgment — or what you also
might call content moderation — in sorting out what expression does and does not appear on the platforms that they own? So it’s just not clear right now what
precisely the First Amendment says about that question.”

Are the Texas and Florida laws identical?
</item 3>

<item 4>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
To explain the Supreme Court’s decision to wade into these issues — and what happens next — TechCrunch spoke to Paul Barrett, NYU adjunct law professor and
Deputy Director of NYU Stern’s Center for Business and Human Rights.

Why is the Supreme Court involved?

These cases actually started in Florida and Texas a few years ago before wending their way to the Supreme Court this year. In those two states, Republican
lawmakers passed parallel laws to control how social media companies operate. In Florida, Governor Ron DeSantis signed Senate Bill 7072 into law in May 2021. In
Texas, House Bill 20 made it through the state legislature and was signed by Governor Greg Abbott in September 2021.

“The reason why these cases are in front of the Supreme Court is actually relatively simple: Florida and Texas were more or less the first out of the gate in
imposing this type of restriction on social media companies,” Barrett explained.
</item 4>

<item 5>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Time will tell how this one plays out but he may find the dead blue bird he’s chosen to hang around his neck, when he purposefully ripped up the standard
content moderation rulebook and opted to channel online chaos, starting to feel like it’s dragging him down.
</item 5>

<item 6>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
“So when the industry sued the states under the First Amendment, these were the first cases that were litigated, so they went up through the court system.”

Both laws made their way through the lower courts after tech industry group NetChoice and the Computer and Communications Industry Association (CCIA) issued
legal challenges against them. That path was complicated and contradictory, which is part of how the case landed in the Supreme Court’s lap:

In both cases, you had federal trial judges who entered injunctions blocking the laws on constitutional grounds. And then you had two different federal
appellate courts — in the case of Florida, the Eleventh Circuit, in the case of Texas, the Fifth Circuit. The two appellate courts clashed… and there was an
explicit conflict between the two federal appellate courts. And that type of conflict is one of the bases that the US Supreme Court uses for deciding when to
take cases.

What does this have to do with the First Amendment?
</item 6>

<item 7>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Under Musk, X has pushed the use of an existing crowdsourced fact-checking feature (rebranded as Community Notes) which essentially seeks to outsource the issue
to a hands-length process of aggregating views from platform users, some of which may be appended as contextual notes to dubious tweets. At the same time the
erratic billionaire has set about removing signals Twitter users were previously able to rely on to help assess the quality of information tweets contained,
such as removing state-affiliated media labels from propaganda outlets and replacing legacy account verifications with pay-to-play.

X was contacted for a response to the EU’s analysis of the ratio of disinformation it’s spreading. It replied with an automated email reading: “Busy now, please
check back later.”
</item 7>

<item 8>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
What’s at stake in the Supreme Court’s landmark social media case The court will wade into First Amendment questions with major implications for platforms

The Supreme Court is poised to decide whether a pair of state laws are allowed to reshape the ability of social media companies to control what does — and
doesn’t — appear on their platforms.

Last week, the Supreme Court decided that it would hear the pair of cases, which revolve around Republicans crafting state-specific laws that order platforms to
keep their hands off of some social media posts. Since the early days of the Trump administration, Republicans have accused social media companies of
deliberately suppressing conservative viewpoints.

While research has not supported these claims, researchers have demonstrated that conservative social media users are disproportionately exposed to political
misinformation, a phenomenon that could explain anecdotal claims of ideologically lopsided enforcement on social platforms.

Whether animated by those perceived differences or the political perks of accusing social platforms of anti-conservative bias, conservative lawmakers in Florida
and Texas passed laws to restrict how those companies are allowed to moderate content.
</item 8>

<item 9>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Aside from forcing platforms to allow some forms of content that would otherwise be disallowed, these laws also seek to force social media companies to provide
users with individualized explanations when their content is removed or restricted. Because this process is largely conducted algorithmically now — generally
with light human intervention or oversight — social media companies might need to reimagine their content moderation systems or hire way more humans to respond
to these incidents. Those kind of adjustments would likely be costly and difficult to scale.

“As you can imagine, that can get kind of onerous when you’re taking down millions and millions of pieces of content a day, and much of that activity of the
vast majority of it is currently being done automatically,” Barrett said. “The idea that a human being would have to go back and explain each time something
came down would be quite a challenge.”
</item 9>

<item 10>
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
“Thanks to the First Amendment, Florida and Texas cannot force websites or social media apps to host hateful content, misinformation and spam, as their deeply
misguided laws would require,” Oregon Senator Ron Wyden, who co-authored Section 230, a law that protects social media companies’ content moderation decisions,
told TechCrunch. “… A ruling in favor of the Texas and Florida laws would create utter chaos, and make many sites worthless to regular users who want to watch a
funny video or see family photos.”

NetChoice President Steve DelBianco also warned that allowing the state laws to go into effect would unleash “a tidal wave of offensive content and hate speech
crashing onto users, creators, and advertisers” that would force Americans to wade through “racial epithets, aggressive homophobia, pornographic material,
beheadings, or other gruesome content” just to use social apps.
</item 10>
=== END ITEM id=b49b08e8e25d9c41 ===

=== ITEM id=6a43cf1c6fc9b298 ===
Gold sentences and facts:
Sentence 0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 0.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 0.1: The defense countered that the FTX founder acted in good faith.
  Fact 0.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 0.3: The defense countered that the FTX founder basically got in over his head.
Sentence 1: It was Bankman-Fried’s decision, she said, as he was the CEO of FTX.
  Fact 1.0: A speaker referred to as 'she' said it was Bankman-Fried's decision.
  Fact 1.1: The speaker said Bankman-Fried was the CEO of FTX.
  Fact 1.2: The speaker said it was Bankman-Fried's decision as he was the CEO of FTX.

Retrieved items:
<item 1>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.
</item 1>

<item 2>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
When he was evaluating the exchange, he thought the sheer volume of users was important, too — at the time, FTX was among the top three biggest exchanges. Plus,
major VC firms had invested, and “they don’t commit hundreds of millions without doing due diligence, checking the books, the accountancy of the firm, going
through several compliance process[es], so that was a vote of confidence for me,” Juilliard said. (Evidently he had not paid attention to the Elizabeth Holmes
trial.)

He also noted FTX’s glossy ads — featuring Gisele Bündchen, for instance — suggested a very high budget. It wouldn’t make sense to spend that much money unless
FTX had very strong financials, Juilliard figured. He opened an account, transferred in both regular money and cryptocurrency, and used the exchange to execute
his plan: buying Bitcoin to sell back in five to ten years at higher prices.
</item 2>

<item 3>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Joe Bankman, Bankman-Fried’s father. He is among the people included in the “small group chat,” the Signal group where the FTX inner circle strategized during
the exchange’s last days.

Barbara Fried’s political action committee, Mind the Gap. It is among the entities that received FTX customer money as political donations. Fried and Bankman
lived together in a Bahamas house purchased for them using FTX investor funds, according to a financial expert called by the prosecution; in a lawsuit filed by
FTX’s bankruptcy estate, Bankman and Fried are alleged to have been given $10 million of FTX customer funds.

Gabe Bankman-Fried’s nonprofit, Guarding Against Pandemics, which got millions from FTX. He was also in a Signal group chat, Donations Processing, that Singh
says was used to coordinate the transfer of FTX and Alameda money to various political causes. In that chat, he can be seen recommending some donations be made
“in the dark” to avoid public relations headaches.
</item 3>

<item 4>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 4>

<item 5>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 5>

<item 6>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
A substantial chunk had been spent on various investments, including in the celebrity-connected venture firm K5, trading firm Modulo (co-founded by one of
Bankman-Fried’s ex-girlfriends), and AI company Anthropic.

Another $413 million had gone directly to Bankman-Fried himself, through his wholly-owned company Paper Bird. “Outflows to insiders,” which suggested Singh’s
not-quite-loans were included, was $192 million; real estate was another $228 million; and the donations to Gabe Bankman-Fried’s Guarding Against Pandemics,
combined with the philanthropy from the FTX Foundation, totaled less than $40 million.

It is easy to lie with words and harder to lie with money.
</item 6>

<item 7>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 7>

<item 8>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 8>

<item 9>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 9>

<item 10>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 10>
=== END ITEM id=6a43cf1c6fc9b298 ===

=== ITEM id=a3f34a151f51a948 ===
Gold sentences and facts:
Sentence 0: As an added bonus, Harris is also the preferred goal-line back, scoring in three of those games.
  Fact 0.0: Harris is also the preferred goal-line back.
  Fact 0.1: Harris scored in three of those games.
Sentence 1: The Wildcats on their first drive put a touchdown on the board a seven play, 61 yard drive, thanks to a 25-yard dash before diving into the end zone by running back Jonah Coleman.
  Fact 1.0: The Wildcats scored a touchdown on their first drive.
  Fact 1.1: The Wildcats' first drive was a seven play drive.
  Fact 1.2: The Wildcats' first drive covered 61 yards.
  Fact 1.3: The Wildcats' first-drive touchdown came thanks to a 25-yard dash.
  Fact 1.4: The 25-yard dash was followed by a dive into the end zone.
  Fact 1.5: Running back Jonah Coleman dove into the end zone.

Retrieved items:
<item 1>
No. 17 Arizona dominates No. 22 Utah, 42-18 (Rivals, 2023-11-18T23:45:32+00:00)
"I don't think the rain effected us at all," Fisch said.

With the rain starting to clear up to start the second half, overtime it was clear that the Wildcats’ defense was holding a cloud over the Utes’ heads. Arizona
got off to a slow start but picked up much of its thunderous momentum after junior safety Gunner Maldonado broke up a pass on fourth and three. Since the
fourth-down stop, the UA logged an interception and another turnover on downs.

Although the defense was surging, the Wildcats not scoring since the first drive of the second quarter — their offense seemed to have lost its touch.

But then “Swag Surfin” by Fast Life Yungstaz came on, signaling the fourth quarter, the quarter in which going into the game, Arizona was outscoring opponents
90-37. The Wildcats on their first drive put a touchdown on the board a seven play, 61 yard drive, thanks to a 25-yard dash before diving into the end zone by
running back Jonah Coleman.
</item 1>

<item 2>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Patriots

22 Jerome Ford, Browns vs. Jaguars

23 Alexander Mattison, Vikings @ Raiders

24 Najee Harris, Steelers vs. Patriots

25 Ezekiel Elliott, Patriots @ Steelers

26 Gus Edwards, Ravens vs. Rams

27 James Cook, Bills @ Chiefs

28 Zack Moss, Colts @ Bengals

29 Keaton Mitchell, Ravens vs. Rams

30 AJ Dillon, Packers @ Giants

31 Dameon Pierce, Texans @ Jets

32 Aaron Jones, Packers @ Giants

33 Clyde Edwards-Helaire, Chiefs vs. BIlls

34 Jerick McKinnon, Chiefs vs. Bills

35 Roschon Johnson, Bears vs. Lions

36 Tyjae Spears, Titans @ Dolphins

37 Kareem Hunt, Browns vs. Jaguars

38 Tyler Allgeier, Falcons vs. Buccaneers

39 Chuba Hubbard, Panthers @ Saints

40 D'Onta Foreman, Bears vs.
</item 2>

<item 3>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
49ers

26 Javonte Williams, Broncos vs. Browns

27 AJ Dillon, Packers @ Lions

28 Jerome Ford, Browns @ Broncos

29 Devin Singletary, Texans vs. Jaguars

30 Kareem Hunt, Browns @ Broncos

31 James Cook, Bills @ Eagles

32 Jaylen Warren, Steelers @ Bengals

33 Najee Harris, Steelers @ Bengals

34 Khalil Herbert, Bears @ Vikings

35 Tyler Allgeier, Falcons vs. Saints

36 Chuba Hubbard, Panthers @ Titans

37 Ezekiel Elliott, Patriots @ Giants

38 Latavius Murray, Bills @ Eagles

39 Roschon Johnson, Bears @ Vikings

40 Joshua Kelley, Chargers vs. Ravens

41 Keaton Mitchell, Ravens @ Chargers

42 Ty Chandler, Vikings vs. Bears

43 Dameon Pierce, Texans vs.
</item 3>

<item 4>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Eagles

15 Kyren Williams, Rams @ Ravens

16 Joe Mixon, Bengals vs. Colts

17 Javonte Williams, Broncos @ Chargers

18 Derrick Henry, Titans @ Dolphins

19 D'Andre Swift, Eagles @ Cowboys

20 Jaylen Warren, Steelers vs. Patriots

21 Alexander Mattison, Vikings @ Raiders

22 Jerome Ford, Browns vs. Jaguars

23 Kenneth Walker III, Seahawks

24 Najee Harris, Steelers vs. Patriots

25 James Cook, Bills @ Chiefs

26 Zack Moss, Colts @ Bengals

27 Keaton Mitchell, Ravens vs. Rams

28 AJ Dillon, Packers @ Giants

29 Ezekiel Elliott, Patriots @ Steelers

30 Aaron Jones, Packers @ Giants

31 Jerick McKinnon, Chiefs @ Bills

32 Tyjae Spears, Titans @ Dolphins

33 Gus Edwards, Ravens vs. Rams

34 Dameon Pierce, Texans @ Jets

35 Roschon Johnson, Bears vs.
</item 4>

<item 5>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
We still like Jaylen Warren better overall, but you can confidently take Najee Harris (vs. Patriots) out of the doghouse for good. Harris has been running with
purpose, and his role in the passing game continues to make him a high-ceiling RB2 in a run-first offense that will be without Kenny Pickett (ankle) this week.
New England might be banged up, but it's still a tough D to get chunk yardage against through the air. Look for Harris to see lots of screens and dump-downs in
a rough-and-tumble Thursday night game in Pittsburgh.

We love Baltimore's Keaton Mitchell (vs. Rams) and think he has the potential to smash Los Angeles at home this week.
</item 5>

<item 6>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I'd rather start Breece Hall, Rhamondre Stevenson, Dameon Pierce and Jaleel McLaughlin (if Javonte Williams sits out).

LAST WEEK: Harris had his first game with over 100 total yards against the Texans and averaged 5.1 yards per rush, just shy of the season-high he technically
had on six carries against the 49ers in Week 1. A 32-yard reception helped him cross the triple-digit barrier.

HARRIS: This might shock you, but Harris is averaging 4.3 yards per run this year with a 12.2% explosive run rate. Would you believe the explosive run rate,
which reflects the frequency of his 12-plus-yard runs, is fifth-best among 44 qualifying running backs?! He's also 13th in avoided tackle rate and top-10 in
yards after contact per carry at 3.53.

TOUCHDOWNS: Harris has zero.
</item 6>

<item 7>
Patriots vs. Steelers FanDuel Picks: NFL DFS lineup advice for Week 14 TNF single-game tournaments (Sporting News, 2023-12-07T21:26:00+00:00)
Expect the RB to be fed a ton. Whether that results in a touchdown remains to be seen, but if anyone is scoring a TD on the Pats roster Thursday night, it's
Elliott.

FLEX RB Najee Harris, Steelers ($14,000). Much like the Patriots, this Steelers offense will want to establish the ground game, and that starts with Harris. It
wasn't long ago that the RB was the subject of major criticism for looking slow and ineffective, but that narrative has flipped in a big way. With an average of
15 carries per game in the past five, we can expect at least that and maybe more from a Pittsburgh team that is starting Mitch Trubisky on Thursday night. As an
added bonus, Harris is also the preferred goal-line back, scoring in three of those games. The Patriots' defense doesn't give up much, making Harris our best
bet for success in an old-fashioned defensive struggle between two of the worst offenses the NFL has to offer.
</item 7>

<item 8>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
At least all of them went to his wideouts. All were in the red zone, none were on deep throws and none were on WR screens.

Sit Him (Lineup Decisions) Projections powered by Sportsline Najee Harris RB PIT Pittsburgh • #22

Age: 25 • Experience: 3 yrs. Weekly Breakdown MATCHUP vs BAL BAL -4 O/U 38 OPP VS RB 12th PROJ PTS 9.3 RB RNK 30th YTD Stats RUYDS 210 REC 4 REYDS 34 TD 0
FPTS/G 7.1 The Steelers' lack of scoring opportunities really crushes the upside for Harris, who admittedly has improved this season and could be on his way to
being a reliable RB2. The Ravens have been pretty good against the run but they can give up some numbers. Harris is a volume-based back who is at best a low-end
starting option given everything that's going on.
</item 8>

<item 9>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
TOUCHDOWNS: Harris has zero. Somehow it gets worse: He has one carry inside the 15-yard line. The Steelers have run the second-fewest red-zone plays (18, one
more than New England) and second-fewest plays inside the 5-yard line (two, one more than New England). The Steelers have played four games! This is nuts.

RAVENS: Have yet to allow a rushing score to a running back this year. They've been otherwise solid against RBs, holding them to 4.0 yards per carry on the
season and roughly league-average in other metrics including missed tackles (15).
</item 9>

<item 10>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
But just three players have averaged over 4.0 yards per carry: Joe Mixon (4.5 yards per run, didn't have 100 total yards, didn't score), Zack Moss (4.1 yards
per run, ran for 122 yards on 30 carries and caught a touchdown) and Pierre Strong (9.8 yards per carry, most of it coming in garbage time last week).

HISTORY: Harris had over 120 total yards and a touchdown in a Week 17 win in Baltimore last year. He also scored on them with exactly 50 total yards against
them in at home in Week 14. He has posted at least 10 non-PPR points in 3 of 4 career games.
</item 10>
=== END ITEM id=a3f34a151f51a948 ===
