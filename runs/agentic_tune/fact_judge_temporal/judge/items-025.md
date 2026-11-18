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

=== ITEM id=c8d2621877f0b807 ===
Gold sentences and facts:
Sentence 0: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 0.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 0.1: The case argues that Google 'siphons off' news publishers' content, their readers and ad revenue through anticompetitive means.
Sentence 1: Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
  Fact 1.0: Google unveiled new hardware.
  Fact 1.1: Google held its annual hardware event this week.
  Fact 1.2: Google is a search and consumer tech giant.
  Fact 1.3: At the event, Google showed off what it has been working on.

Retrieved items:
<item 1>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Bing
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Bing
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> monopoly power
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Bing

Entities: monopoly power, FTC-Facebook case, Bing, Apple, WhatsApp, TechCrunch, data, Google antitrust case, 5 things we learned so far about the Google
antitrust case, DOJ, FTC, FTC lawsuit against Amazon, anticompetitive and unfair strategies, DOJ investigation of Apple, Apple's policy for third-party apps on
its devices
</item 1>

<item 2>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Meagan Bellshaw
- U.S. Justice Department -PARTICIPATED_IN-> U.S. Justice Department lawsuit against Google
- Meagan Bellshaw -WORKS_FOR-> Justice Department
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. Justice Department
- U.S. Justice Department -PARTICIPATED_IN-> 2020 Justice Department civil antitrust suit over search and search advertising

Entities: Meagan Bellshaw, U.S. Justice Department, TechCrunch, Pichai, Nitin, homepage market share, search market share, Google antitrust case, 5 things we
learned so far about the Google antitrust case, Google, The Verge
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

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- 5 things we learned so far about the Google antitrust case -MENTIONS-> FTC-Facebook case
- Amit Mehta -PARTICIPATED_IN-> US v Google trial

Entities: Amazon antitrust lawsuit, FTC-Facebook case, Google antitrust lawsuit with Match Group, Google trial with Epic Games, Anticompetitive and unfair
strategies to maintain monopoly power, Vertical search, Pre-installs, Third-party apps, Commission structure, More from the US v Google trial: vertical search,
pre-installs and the case of Firefox/Yahoo, US v Google trial, Yahoo, Firefox, anticompetitive behavior, Google Play
</item 3>

<item 4>
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

Graph facts:
- Grey -WORKS_FOR-> DuckDuckGo
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> DuckDuckGo
- Is Google Search better than the rest? And is that fair? -MENTIONS-> DuckDuckGo
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> antitrust

Entities: third-party developers, DuckDuckGo, antitrust, Bing, Microsoft, Apple, monopoly, TechCrunch, data, Google antitrust case, Brave, Microsoft trial,
anticompetitive conduct, APIs, computer manufacturers
</item 4>

<item 5>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Bellshaw
- Microsoft -MENTIONS-> cloud gaming
- Microsoft -MENTIONS-> device-agnostic ecosystem
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Microsoft
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Microsoft

Entities: Bellshaw, Microsoft, TechCrunch, Pichai, Google antitrust case, message exchange, litigation hold, personnel matter, Internet Explorer, David
Drummond, Internet Explorer became the default search engine, Drummond’s letter, default settings, 5 things we learned so far about the Google antitrust case,
Google
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

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- Amit Mehta -SAID-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta
- 5 things we learned so far about the Google antitrust case -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: Amit Mehta, APIs, third-party developers, More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google
trial, Bing, TechCrunch, U.S. government, DOJ, antitrust law, Google, search, advertising
</item 6>

<item 7>
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

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, TechCrunch, Pichai, Google antitrust case, Pichai's testimony, lucrative search agreement, systematically destroying chats, 5 things we learned
so far about the Google antitrust case, DOJ, Google, CNBC, Cook
</item 7>

<item 8>
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

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Sundar Pichai
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sundar Pichai
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Sundar Pichai
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Sundar Pichai
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Sundar Pichai

Entities: Sundar Pichai, Department of Justice, antitrust, Apple, online search, TechCrunch, New York, California, U.S. government, digital advertising, U.S.
government’s antitrust trial against Google, 2020 antitrust claim, January filing, eight states, 5 things we learned so far about the Google antitrust case
</item 8>

<item 9>
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

Graph facts:
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Chrome
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Chrome
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> Chrome
- ‘Mass breach of privacy’: TikTok under fire for tracking users online -MENTIONS-> Chrome
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Chrome

Entities: Chrome, Satya Nadella, DuckDuckGo, Bing, The Information, Microsoft, Apple, TechCrunch, Tim Cook, Pichai, Google antitrust case, 5 things we learned
so far about the Google antitrust case, Google, iOS, iPhone
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 10>
=== END ITEM id=c8d2621877f0b807 ===

=== ITEM id=9d5456af75145bba ===
Gold sentences and facts:
Sentence 0: Erik ten Hag is a top coach from Ajax but they need a top blueprint not only for now but for the next five years, and work towards that, and sometimes on that journey there are hard moments when you don't get the right results, even though you're doing the right things.
  Fact 0.0: Erik ten Hag is a top coach from Ajax.
  Fact 0.1: Ten Hag's club needs a top blueprint not only for now but for the next five years.
  Fact 0.2: The club should work towards that blueprint.
  Fact 0.3: On that journey there are sometimes hard moments when you don't get the right results.
  Fact 0.4: Sometimes you are doing the right things even though you don't get the right results.
Sentence 1: Manchester United are out of Europe altogether after a limp display against Bayern at Old Trafford, where Kingsley Coman sealed a 1-0 win for Thomas Tuchel's side.
  Fact 1.0: Manchester United are out of Europe altogether after a limp display against Bayern.
  Fact 1.1: Manchester United's display against Bayern was at Old Trafford.
  Fact 1.2: Kingsley Coman sealed a 1-0 win at Old Trafford.
  Fact 1.3: Kingsley Coman's goal sealed a 1-0 win for Thomas Tuchel's side.

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
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/IcKGdmJv4O — Optus Sport (@OptusSport) December 23, 2023

Manchester United legend Peter Schmeichel lashed the team’s lack of a plan and blasted them for looking slow against the Hammers.

“It was just in many ways irrelevant. There was no plan to it,” he said. “West Ham got an easy win which is a little bit silly to say.

“When I look at the team, I don’t know what the identity of the team is. What is the style of play, what’s the plan?

Graph facts:
- VAR apology list in Premier League: Updated list of PGMOL apologies for referee mistakes in English football -MENTIONS-> Spurs
- NBA Rookie of the Year Odds - Wembanyama the early favorite -MENTIONS-> Spurs
- Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see -MENTIONS-> Spurs
- Gregg Popovich -WORKS_FOR-> Spurs
- Spurs -PARTICIPATED_IN-> Spurs vs Brighton

Entities: Spurs, Manchester United, West Ham, Ange-ball, Peter Schmeichel, Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays
off again as Spurs sink Everton, Optus Sport, Everton, The Roar | Sports Writers Blog, Ten Hag
</item 2>

<item 3>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Old Trafford (and the training ground)

To find an apt shorthand for the lurching ship that is United, end the search at the roof of their stadium where on Manchester’s (many) wet days rain leaks
through. Old Trafford remains a storied venue but as it continues to crumble the tale of disrepair has become a central narrative in its recent history. As
Premier League titles have become scarce (zero in a decade) the tale of neglect of a ground that dates to 1910 is emblematic of the Glazers’ quasi‑absentee
proprietorship. Ratcliffe will invest $300m (£237m) into infrastructure but this is small beer when considering the multibillion-pound facelift or new venue
required. There is also the cramped Carrington base: as on Sir Matt Busby Way, the footprint is large at the training ground but the configuration and
facilities are ageing – badly.

Graph facts:
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Jean-Claude Blanc
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Ineos Sport
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> transfer policy
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Murtough
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Aaron Ramsey

Entities: From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray, Carrington, Sir Matt Busby Way, Premier League, infrastructure,
Manchester United, Manchester, Old Trafford, Ratcliffe, The Guardian, Glazers, Ten Hag
</item 3>

<item 4>
Graph query row:
article: Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United
published_on: 2023-10-16T21:00:00+00:00
publisher: talkSPORT
passages: ['Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)\nJim Ratcliffe’s move to Monaco in 2020 may have been bad news for the British taxpayer, but not for Manchester United fans.\n\nThe
chemical billionaire looks set to buy a 25 per cent stake in the Premier League giants after fellow bidder, Sheikh Jassim, withdrew his offer.\n\n6 Ratcliffe
could still have a big say at United Credit: getty\n\n6 His wisest move could be to land his old neighbour Mitchell\n\nThat news hasn’t been too warmly welcomed
by United supporters, who would still have to put up with the hugely unpopular Glazer family in control of the club.\n\nIt could, though, mean some sort of
progress, if the Manchester born Ratcliffe follows through with his vision to revolutionise the club.\n\nShould Ratcliffe come in, talkSPORT understands that he
would take charge of footballing matters, something the Glazers have been accused of having little to no interest in.', 'Paul Mitchell signed Tottenham and
Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport, 2023-10-16T21:00:00+00:00)\nAlready a top tier European
owner of OGC Nice, there has been some success, but nothing compared to their French riviera neighbours, where Ratcliffe appears set to look for
inspiration.\n\nPaul Mitchell recently left the principality of Monaco, but he could well be back there in meetings with Ratcliffe who is planning to appoint a
sporting director.\n\nMitchell fulfilled that role at Monaco after previous stints at Leipzig, Tottenham, Southampton and MK Dons, becoming one of football’s
most respected executives.\n\nA former player in England’s lower leagues, Mitchell was previously linked with Chelsea and Liverpool, and much like Ratcliffe,
he’s also from Greater Manchester.\n\nStarring mainly for Wigan and MK Dons, it was with the latter where Mitchell was moved upstairs, and made his first big
move by spotting future England regular Dele Alli after being named their head of recruitment in 2010.', "Paul Mitchell signed Tottenham and Chelsea superstars,
fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport, 2023-10-16T21:00:00+00:00)\nTwo years later he joined Southampton in the
same role, and that’s where he really began to show off, landing the Saints a number of players who would not only go on to become Champions League regulars,
but make the club a fortune and ensure a Premier League best finish.\n\nShowcasing his skills in casting his scouting net far and wide, Sadio Mane, Dusan Tadic,
Graziano Pelle, Dejan Lovren and Victor Wanyama all went on to secure Southampton a bright future, and Mitchell a promotion to Tottenham in 2014.\n\n6 Mane was
one of Mitchell's early successes\n\n6 But there's been many more Credit: Getty\n\nThe north London side had just unsuccessfully splurged their world record
£85million from Real Madrid for Gareth Bale, and it looked like Mitchell couldn’t have picked a worse time.", 'Paul Mitchell signed Tottenham and Chelsea
superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport, 2023-10-16T21:00:00+00:00)\nOnly leaving the side to reach the
pinnacle of his job, finally stepping up from a head of recruitment to become sporting director at Monaco, Mitchell showed just how big picture he is with an
interview in 2021.\n\n“I want to continue to drive standards in each and every department,” he said. “Psychology, sports science, nutrition, medicine, the way
we travel, the way we refuel.\n\n“Everything, every element I think we need to keep pushing to its maximum to try and be the best we possibly can.”\n\nHaving
worked in football management for over a decade, Mitchell’s career has always been about the long term for his club’s, with the young signings he’s made often
only paying off after he’s left.\n\nOne in particular was 2022/23 Premier League top scorer Son, who perfectly sums up his philosophy.\n\n“I took quite a lot of
criticism in my first year at Tottenham for some of the decisions and Sonny was kind of at the centre of that,” he explained.', 'Paul Mitchell signed Tottenham
and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport, 2023-10-16T21:00:00+00:00)\n“Sometimes players need
time, they’re human beings, they need to settle, Sonny had a great track record at Hamburg and Leverkusen but there was this narrative that started to grow
internally and externally that he wasn’t a good finisher.\n\n6 Mitchell is becoming an increasingly recognisable face\n\n“I remember telling a coach ‘you’re
wrong, his history shows in two different clubs he’s a very astute finisher off both sides’.”\n\nMitchell remains good friends with Rangnick from their time
together at Leipzig, and last season made headlines when he was spotted at a youth team game while the German was interim coach at Man United.\n\nSo astute is
Mitchell in his observations, he even laid out a blueprint for United\'s success while at Monaco.\n\n“I think context is important to measure what Manchester
United\'s current situation is, but I think they do need to get to the point of really putting down what Manchester United needs to be in the modern game,” he
explained.\n\n"We are a long time from the Sir Alex Ferguson era.']
key_entities: [{'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'Champions League'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'name': 'Premier League'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Southampton'}, {'labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'name': 'Chelsea'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Manchester United'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Place'], 'name': 'England'}, {'labels': ['__KGBuilder__', '__Entity__', 'Place'], 'name': 'Manchester'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Daniel Levy'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jim Ratcliffe'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'name': 'Glazer family'}, {'labels': ['__KGBuilder__', '__Entity__', 'Place'], 'name': 'Monaco'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Topic'], 'name': 'footballing matters'}, {'labels': ['__KGBuilder__', '__Entity__', 'Event'], 'name': "Jim Ratcliffe's move to Monaco"},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Paul Mitchell'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Dele Alli'},
{'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'OGC Nice'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Monaco'},
{'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Leipzig'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'MK Dons'},
{'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Wigan'}, {'labels': ['__KGBuilder__', '__Entity__', 'Place'], 'name': 'French Riviera'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Sadio Mane'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Dusan Tadic'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Dejan Lovren'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Victor Wanyama'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Gareth Bale'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Sonny'}, {'labels':
['__KGBuilder__', '__Entity__', 'Place'], 'name': 'Leverkusen'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Rangnick'}, {'labels':
['__KGBuilder__', '__Entity__', 'Event'], 'name': 'youth team game'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Real Madrid'},
{'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'Premier League'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Ratcliffe'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Sir Alex Ferguson'}, {'labels': ['__KGBuilder__', '__Entity__', 'Place'], 'name': 'Hamburg'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Sheikh Jassim'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Graziano Pelle'},
{'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Liverpool'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name':
'Tottenham'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Son'}, {'labels': ['__KGBuilder__', '__Entity__', 'Place'], 'name': 'Greater
Manchester'}]
connected_articles: []
</item 4>

<item 5>
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

Graph facts:
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Jean-Claude Blanc
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Ineos Sport
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> transfer policy
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Murtough
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Aaron Ramsey

Entities: From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray, Greig Cowie, Manchester City, Manchester United, Google, Ratcliffe,
The Guardian, the Glazers, Liverpool, Shutterstock, Football Daily, Google reCaptcha, Ten Hag, communications
</item 6>

<item 7>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
Garnacho scored with an incredible overhead kick and Mainoo impressed on what was his full Premier League debut, which could earn him a start in Turkey and an
education in European football.

Around them are players who have succeeded on hostile nights. But if they thought Goodison Park was unwelcoming, they will be in for a surprise on Wednesday and
will need to find communal coping mechanisms. Fail to do so and they will just have to live and learn for next time.

Graph facts:
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Champions League
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Alex Ferguson
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Eric Cantona
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> riot police
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Cerberus

Entities: Manchester United face Galatasaray with high hopes but bad memories, Galatasaray, Turkey, Mainoo, European football, Garnacho, Manchester United,
Goodison Park, Premier League, The Guardian
</item 7>

<item 8>
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

Graph facts:
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> Champions League
- Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United -MENTIONS-> Champions League
- Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage -MENTIONS-> Champions League
- Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world -MENTIONS-> Champions League
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Champions League

Entities: Champions League, Manchester United, FA Cup, The Guardian, City, Barney Ronay, Ten Hag’s Manchester United stand up and die at club stuck in the past
tense | Barney Ronay, Europe, Ten Hag
</item 8>

<item 9>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

“I think that it is important sometimes that you have to go through those moments to realise what it takes. There are elements that happen in a football match
and elements that happen in the buildup to a football match, so you have to experience all those things if you want to be successful.”

The win against Everton on Sunday was defined by two teenagers, Alejandro Garnacho and Kobbie Mainoo.

Graph facts:
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Champions League
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Alex Ferguson
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Eric Cantona
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> riot police
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Turkey

Entities: Manchester United face Galatasaray with high hopes but bad memories, Galatasaray, Kobbie Mainoo, win against Everton, Alejandro Garnacho, Manchester
United, The Guardian, Everton, football
</item 9>

<item 10>
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

Graph facts:
- Everton players and fans will be fired up by points penalty, asserts Sean Dyche -MENTIONS-> Chelsea
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Chelsea
- Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn -MENTIONS-> Chelsea
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Chelsea
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Chelsea

Entities: Chelsea, Manchester United, Daniel Levy, Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for
Manchester United, Paul Mitchell, Monaco, Leipzig, Sonny, Leverkusen, Rangnick, youth team game, talkSPORT, Sir Alex Ferguson, Hamburg, Tottenham
</item 10>
=== END ITEM id=9d5456af75145bba ===

=== ITEM id=4869f03b811dd2e5 ===
Gold sentences and facts:
Sentence 0: Despite the suspension, Jim Harbaugh held his usual weekly news conference Monday, though he mostly avoided addressing the only topic on anyone's mind.
  Fact 0.0: Jim Harbaugh was suspended.
  Fact 0.1: Despite the suspension, Jim Harbaugh held his usual weekly news conference on Monday.
  Fact 0.2: Jim Harbaugh mostly avoided addressing the only topic on anyone's mind.
Sentence 1: It would be Harbaugh's second suspension of the season, as the Wolverines self-imposed a three-game ban on Harbaugh to start the season amid an NCAA investigation that found Harbaugh lied to investigators.
  Fact 1.0: The suspension in question would be Harbaugh's second suspension of the season.
  Fact 1.1: The Wolverines self-imposed a three-game ban on Harbaugh to start the season.
  Fact 1.2: The three-game ban was imposed amid an NCAA investigation.
  Fact 1.3: The NCAA investigation found that Harbaugh lied to investigators.
Sentence 2: Harbaugh reportedly watched the game from the team hotel, and now the leading piece of drama returns.
  Fact 2.0: Harbaugh reportedly watched the game from the team hotel.
  Fact 2.1: The leading piece of drama returns now.

Retrieved items:
<item 1>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Rice on Saturday and at South Florida on Nov. 25.

I am extending an open invitation to Jim Harbaugh to join me on the sidelines for the next two Charlotte football games. It would be my honor to stand next to
you again as we did for three years — Biff poggi (@BiffPoggi) November 17, 2023

Thursday, Nov. 16

Michigan accepts Harbaugh's three-game suspension

Michigan has dropped its litigation against the Big Ten over Harbaugh's three-game suspension, and it will accept the punishment levied against the head coach.

From Michigan:

"This morning, the University, Coach Harbaugh, and the Big Ten resolved their pending litigation. The Conference agreed to close its investigation, and the
University and Coach Harbaugh agreed to accept the three-game suspension. Coach Harbaugh, with the University's support, decided to accept this sanction to
return the focus to our student-athletes and their performance on the field.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, sign-stealing scandal, Big Ten, Biff Poggi, Charlotte, South Florida, Jim Harbaugh, three-game suspension, pending litigation, NCAA cheating
investigation, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Sporting News, Rice,
NCAA
</item 1>

<item 2>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
The Conference has confirmed that it is not aware of any information suggesting Coach Harbaugh's involvement in the allegations. The University continues to
cooperate fully with the NCAA's investigation.

From the Big Ten:

"The Big Ten Conference's commitment to student-athletes, sportsmanship and the commissioner's duty to protect the integrity of competition will never waver.
Today's decision by the University of Michigan to withdraw its legal challenge against the conference's November 10th notice of disciplinary action is
indicative of the high standards and values that the conference and the university seek to uphold, the University of Michigan is a valued member of the Big Ten
Conference and the conference will continue to work cooperatively with the university and the NCAA during this process."

Monday, Nov. 13

Harbaugh says he'll state his case at Friday hearing

Despite the suspension, Jim Harbaugh held his usual weekly news conference Monday, though he mostly avoided addressing the only topic on anyone's mind.

Graph facts:
- Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Jim Harbaugh
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Jim Harbaugh
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> sign-stealing scandal
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Jim Harbaugh
- Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game -MENTIONS-> Jim Harbaugh

Entities: sign-stealing scandal, Jim Harbaugh, Big Ten Conference, University of Michigan, University of Michigan legal challenge withdrawal, University of
Michigan notice of disciplinary action, Jim Harbaugh news conference, Jim Harbaugh hearing, cheating investigation, Michigan sign-stealing scandal news: Latest
updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Sporting News, NCAA investigation, NCAA
</item 2>

<item 3>
Graph query row:
article: Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti
published_on: 2023-11-11T22:50:13+00:00
passage: Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
The Big Ten announced Michigan coach Jim Harbaugh will be prohibited from coaching on the sideline for the remainder of the regular season.

At least, in theory. The Friday ruling is the next chapter in an NCAA investigation into Michigan for in-person scouting and sign stealing in a scandal that led
to the resignation of staffer Connor Stallions on Nov. 3. Harbaugh will be allowed to coach during the week but will not be able to be on the sideline. Michigan
did not get a ruling on a temporary restraining order ahead of the showdown against No. 10 Penn State.

The Wolverines still won 24-15 under interim coach Sherrone Moore, but the fallout from the scandal continues to have a far-reaching impact on the Big Ten,
Michigan and college football beyond this year's College Football Playoff race.

A look at the winners and losers from the scandal to date:

Michigan scandal winners and losers

Winners

Tony Petitti

Petitti certainly did not hit Michigan with a "slap on the wrist."
</item 3>

<item 4>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
On the field, Michigan is in the midst of what could be a historic season. Off the field, the the focus has rarely been on the team's success.

An investigation into Michigan's alleged sign-stealing scheme, centered on now-fired staffer Connor Stalions, continues to evolve by the hour, with one update
after another adding another wrinkle to the saga that began in October. The Big Ten suspended coach Jim Harbaugh for three games on Friday with the team on a
plane en route to Penn State, and although Michigan had hoped to block the suspension, the Wolverines will be without their coach on the sidelines in Happy
Valley.

The Sporting News is tracking the latest updates from the Michigan sign-stealing scandal. Follow along with the newest developments below.

MORE ON MICHIGAN SIGN-STEALING:

Michigan sign-stealing scandal updates

Friday, Nov.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Penn State, Big Ten, Happy Valley, Michigan sign-stealing scheme investigation, Jim Harbaugh suspension, Jim Harbaugh, Michigan
sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Connor Stalions, Sporting News, sign-stealing
scheme, NCAA
</item 4>

<item 5>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
OFFICIAL: Jim Harbaugh suspended for the remaining three regular season games.



"Michigan has been found in violation of the Big Ten Sportsmanship Policy for conducting an impermissible, in-person scouting operation over multiple years..."
https://t.co/htnUkshmnF — Chris Vannini (@ChrisVannini) November 10, 2023

Jim Harbaugh joins Michigan for trip to State College

According to the sleuths on the ground in Ann Arbor, Harbaugh is on the team bus and heading to the airport with his team as Michigan awaits word on a potential
suspension.

Jim Harbaugh boarding the bus pic.twitter.com/CWBJbwEEhi — Austin Meek (@byAustinMeek) November 10, 2023

Michigan still awaiting word from Big Ten

According to ESPN's Adam Rittenberg, Michigan still hasn't heard about an impending punishment from the Big Ten as its 1 p.m. ET flight looms.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Big Ten, ESPN, Jim Harbaugh suspension, Jim Harbaugh, Chris Vannini, Big Ten Sportsmanship Policy, impermissible, in-person scouting
operation, Austin Meek, Ann Arbor, State College, Adam Rittenberg, NCAA cheating investigation, Michigan sign-stealing scandal news: Latest updates on possible
Jim Harbaugh suspension from NCAA cheating investigation, Sporting News
</item 5>

<item 6>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Nicole Auerbach (@NicoleAuerbach) November 11, 2023

Harbaugh hearing coming down to wire before Penn State game

ESPN's Pete Thamel reported it's unclear Harbaugh will be granted a hearing in time to be able to coach against Penn State. With kickoff scheduled for noon ET,
Thamel said "the drama's going to go right up 'till noon."

Latest from ⁦@CollegeGameDay⁩ on Jim Harbaugh’s TRO legal filing. Will he coach Michigan at Penn State today? pic.twitter.com/bz3R1eYxI9 — Pete Thamel
(@PeteThamel) November 11, 2023

The case is reportedly in the hands of Judge Carol Kunhke, a Michigan alumnus.

Harbaugh was not on the Michigan team bus with other coaches and players when it arrived at Beaver Stadium Saturday morning.

Michigan players and coaches arrive. No Jim Harbaugh on the bus.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Nicole Auerbach, Penn State, Michigan sign-stealing scandal, Beaver Stadium, ESPN, Jim Harbaugh, NCAA cheating investigation, Carol Kunhke,
Harbaugh hearing, TRO legal filing, Penn State game, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating
investigation, Pete Thamel, Sporting News
</item 6>

<item 7>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Will Harbaugh be on the field before the end of the regular season? The Big Ten made a Friday ruling that will prohibit Harbaugh from being on the field for the
final three regular-season games as a result of the in-person scouting and sign-stealing scandal that led to the resignation of staffer Conor Stallions. ESPN
reported there will be a temporary restraining order hearing on Nov. 17 that will answer that question.

On the field, Michigan answered the question about its College Football Playoff. Sort of. The Wolverines proved they could beat Penn State (8-2, 5-2) in a
20-yard box. Neither team completed a pass of 20 yards or more. Michigan trailed 3-0 after the first quarter. The Wolverines had eight carries for 10 yards, and
Penn State's Abdul Carter set the tone with a jarring hit on Blake Corum that nearly resulted in a fumble.

Michigan took control with the running game.

Graph facts:
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation -MENTIONS-> Michigan
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan

Entities: Michigan, Penn State, temporary restraining order hearing, Big Ten, Blake Corum, ESPN, Jim Harbaugh, Ohio State, Conor Stallions, College Football
Playoff, Sporting News, in-person scouting and sign-stealing scandal, Abdul Carter, Michigan-Penn State game, Michigan didn't need Jim Harbaugh against Penn
State, but Ohio State is a different game
</item 7>

<item 8>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
pic.twitter.com/Qm9iehYVTR — Pete Thamel (@PeteThamel) November 11, 2023

Harbaugh will not be on the sidelines vs. Penn State

Michigan filed a temporary restraining order against the Big Ten on Friday night, hoping to block Harbaugh's suspension. But there will not be a ruling before
the Wolverines' game vs. Penn State, with the hearing scheduled for Friday, Nov. 17.

No ruling on Michigan’s TRO filing today, so Jim Harbaugh will not coach vs. Penn State.



Statement from the school: “We look forward to presenting our case next week where we intend to demonstrate that the Big Ten has not acted legally or fairly.”

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, Penn State, Big Ten, Jim Harbaugh suspension, Jim Harbaugh, NCAA cheating investigation, Michigan game vs. Penn State, temporary restraining
order filing, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Pete Thamel, hearing,
Sporting News, NCAA
</item 8>

<item 9>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Although the Big Ten has closed its investigation, we are continuing to cooperate with the NCAA as it moves forward with its ongoing investigation. Consistent
with our commitment to integrity, we will continue to take the appropriate actions, including disciplinary measures, based on information we obtain. Earlier
today, Michigan Athletics relieved Chris Partridge of his duties as a member of the Michigan Football staff. Due to employee privacy laws, we are unable to
comment further."

Because Michigan and Harbaugh on Thursday dropped their litigation against the Big Ten, he will not be on Michigan's sideline this week against Maryland and
next week against Ohio State to end the regular season.

Former Wolverines assistant and Harbaugh confidant Biff Poggi on Friday offered his former boss an invitation. Poggi, in his first year as Charlotte's head
coach, asked Harbaugh to join the 49ers for their games vs. Rice on Saturday and at South Florida on Nov. 25.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, sign-stealing scandal, Maryland, Big Ten, Chris Partridge, Michigan Football, Biff Poggi, Charlotte, South Florida, Jim Harbaugh, NCAA
cheating investigation, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation, Michigan
Athletics, Ohio State, Sporting News
</item 9>

<item 10>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
ET flight looms.

Michigan's administration has not heard from the Big Ten as of this hour, per sources. The team will begin its preparations to leave for Penn State around 1
p.m. ET. Really going down to the wire with any discipline, which, as several sources noted, isn't really fair to players. — Adam Rittenberg (@ESPNRittenberg)
November 10, 2023

Harbaugh still expected to fly to State College

Dinich reported earlier on Friday that punishment to Michigan could come around the time the team is expected to get on a plane and fly to Pennsylvania for its
game against Penn State. But regardless of when the punishment is announced, Harbaugh's attorney, Tom Mars, reportedly told ESPN the coach will be headed to
State College.

"I'm confident he's going to be on that plane one way or another," Mars reportedly told ESPN.

Graph facts:
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> Michigan
- After a Stunning Snub, College Football Is Facing an Identity Crisis -MENTIONS-> Michigan
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Michigan
- USC basketball futures odds to win Pac-12, NCAA tourney ahead of Bronny James collegiate debut -MENTIONS-> Michigan
- Michigan conquers Ohio State again, but the story is far from over for 2023 Wolverines -MENTIONS-> Michigan

Entities: Michigan, sign-stealing scandal, Penn State, Big Ten, suspension, Jim Harbaugh, State College, game against Penn State, Tom Mars, Dinich, Adam
Rittenberg, cheating investigation, Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation,
ESPN, Sporting News
</item 10>
=== END ITEM id=4869f03b811dd2e5 ===
