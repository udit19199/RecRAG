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

=== ITEM id=8e97c217493f6637 ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury that “all of that, all of it, was built on lies.”
Sentence 1: FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges in relation to the FTX collapse.
  Fact 1.0: Gary Wang is an FTX co-founder.
  Fact 1.1: Gary Wang is the former CTO of FTX.
  Fact 1.2: Caroline Ellison is the former CEO of Alameda Research.
  Fact 1.3: Gary Wang pleaded guilty in December 2022 to federal criminal charges in relation to the FTX collapse.
  Fact 1.4: Caroline Ellison pleaded guilty in December 2022 to federal criminal charges in relation to the FTX collapse.

Retrieved items:
<item 1>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, FTX trial, The FTX trial is bigger than Sam Bankman-Fried, extensive media tour, interviews, Andrew Ross Sorkin, Good Morning
America, The New Yorker, FTX, Sam Bankman-Fried, The Verge
</item 1>

<item 2>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Witness Adam Yedidia answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in
New York City, October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Two of Sam Bankman-Fried's former friends from MIT, who also worked at crypto exchange FTX while living with the company's founder in the Bahamas, took the
stand in a Manhattan courtroom this week to testify against their former classmate, confidant, and boss — a man who allegedly ran a crypto empire that defrauded
thousands of customers out of billions of dollars. Gary Wang, the lesser-known co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday,
"Did you commit financial crimes while working at FTX?" "Yes," responded Wang.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, Adam Yedidia, New York City, Bahamas, Jane Rosenberg, FTX, Nicolas Roos, Sam Bankman-Fried fraud trial, Federal Court in New York City,
financial crimes, Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court, Sam Bankman-Fried, Manhattan, Reuters,
CNBC
</item 2>

<item 3>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Securities and Exchange Commission
- Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison -MENTIONS-> Securities and Exchange Commission
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> misappropriation of customer funds

Entities: Securities and Exchange Commission, misappropriation of customer funds, Sam Bankman-Fried trial, wire fraud, Ellison, FTX, Sam Bankman-Fried,
conspiracy to take FTX customer funds, fake balance sheets, How is it still getting worse for Sam Bankman-Fried?, Genesis, The Verge
</item 3>

<item 4>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Bankman-Fried faces seven federal charges, including wire fraud, securities fraud and money laundering, that could put him in prison for the rest of his life.
Thus far, Bankman-Fried, 31, has remained mostly quiet in court intently listening to witnesses and at times writing notes to his attorneys. But as Wang
testified against him, Bankman-Fried looked visibly upset, shifting his gaze from his former friend to the ground, and at one point putting his head in his
hands.

Sam Bankman-Fried listens as Assistant U.S. Attorney Nicolas Roos questions Gary Wang during Bankman-Fried's fraud trial over the collapse of FTX, the bankrupt
cryptocurrency exchange, at Federal Court in New York City, U.S., October 6, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Wang, 30, was technology chief for FTX, which spiraled into bankruptcy in November.
</item 4>

<item 5>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> FTX collapse
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> SBF’s trial
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> The New York Times
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> SBF
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Series C round

Entities: FTX collapse, SBF’s trial starts soon, but how did he — and FTX — get here?, Damian Williams, Gary Wang, Caroline Ellison, Bankman-Fried’s trial, Wang
and Ellison pleaded guilty to federal criminal charges, Bankman-Fried pleaded not guilty, financial fraud, conspiracy to commit money laundering, conspiracy to
misuse customer funds, civil penalties, federal criminal charges, Alameda Research, TechCrunch
</item 5>

<item 6>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Sam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S judge revoked his bail, at a courthouse in New York,
U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

$65 billion line of credit

For months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on him. Both
pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into Friday, was
given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.

U.S.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Ellison
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Ellison
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> Ellison

Entities: Ellison, New York, Jane Rosenberg, FTX, U.S. attorney's office, courthouse, pleading guilty, Wang's testimony, Sex, Signal messages, and sabotaging
FTX's code: SBF execs and Bahamas roommates tell all in court, Wang, hearing, Sam Bankman-Fried, Reuters, Manhattan, CNBC
</item 6>

<item 7>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Gary Wang

Entities: Gary Wang, Securities and Exchange Commission, misappropriation of customer funds, Alameda, Christian Everdell, spot margin program, Sam Bankman-Fried
was a terrible boyfriend, FTX, Sam Bankman-Fried, trial, fraud, The Verge
</item 7>

<item 8>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Ellison is expected to take the stand under a similar arrangement.

U.S. District Judge Lewis Kaplan presides as Gary Wang testifies during the fraud trial of Sam Bankman-Fried over the collapse of FTX, the bankrupt
cryptocurrency exchange, at Federal Court in New York City, U.S., October 6, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

Born in China, Wang moved to the U.S. at age 7, and grew up in Minnesota before going to the Massachusetts Institute of Technology to study math and computer
science. He worked at Google after college. Wang, who first met Bankman-Fried during high school at a summer camp, owned 10% of Alameda, while his boss owned
the other 90%. Wang told the court about the advantages that Alameda received by having code baked into FTX's software that allowed special access to the crypto
exchange.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, New York City, Lewis Kaplan, Alameda, Minnesota, cryptocurrency, Signal, Bahamas, Ellison, China, Jane Rosenberg, FTX, Sex, Signal
messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court, FTX software, fraud trial of Sam Bankman-Fried
</item 8>

<item 9>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
That was before he realized FTX had stolen customer funds — at which point, he resigned.

Gary Wang, the co-founder of FTX and Alameda Research and another college roommate. He took the stand after Yedidia and sang like a canary about the secret
privileges he’d created for Alameda in the FTX code.

Nishad Singh, a longtime family friend, who copped to what sounded an awful lot like classic embezzlement. He said he’d backdated transactions on FTX to help
create doctored balance sheets to show to investors. He pleaded guilty to six charges, including wire fraud, money laundering, and conspiracy counts around his
political donations.

Caroline Ellison, Bankman-Fried’s ex-girlfriend. She told us about the seven phony balance sheets she’d created. She was clear: Alameda took customer funds;
Bankman-Fried knew about it and even invested them against her advice.

Joe Bankman, Bankman-Fried’s father.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, Caroline Ellison, Bankman-Fried, Alameda Research, wire fraud, Nishad Singh, embezzlement, money laundering, conspiracy, political
donations, In the end, the FTX trial was about the friends screwed along the way, Joe Bankman, FTX trial, FTX, Yedidia
</item 9>

<item 10>
Graph query row:
article: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
publisher: TechCrunch
relevant_connections: [{'entity': 'FTX Ventures', 'relationship': 'MENTIONS'}, {'entity': 'FTX collapse', 'relationship': 'MENTIONS'}, {'entity': 'Gary Wang',
'relationship': 'MENTIONS'}, {'entity': 'Caroline Ellison', 'relationship': 'MENTIONS'}, {'entity': 'FTX', 'relationship': 'MENTIONS'}, {'entity': 'Sam
Bankman-Fried', 'relationship': 'MENTIONS'}]
</item 10>
=== END ITEM id=8e97c217493f6637 ===

=== ITEM id=7d8415bb17a18f4b ===
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
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, Go Big in Europe, Mitchell Baker,
Yahoo, France, Mozilla, Firefox, monopoly, TechCrunch, Google, Germany, search, Europe, competition
</item 1>

<item 2>
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
</item 2>

<item 3>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Chrome
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Chrome
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> Chrome
- ‘Mass breach of privacy’: TikTok under fire for tracking users online -MENTIONS-> Chrome
- The best VPN services for 2023 -MENTIONS-> Chrome

Entities: Chrome, Mozilla, Firefox, technology industry, Yahoo!, Is Google Search better than the rest? And is that fair?, Bing, Microsoft, Google, Walker, The
Age, lawsuit, Windows
</item 3>

<item 4>
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

Graph facts:
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Chrome
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Chrome
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> Chrome
- ‘Mass breach of privacy’: TikTok under fire for tracking users online -MENTIONS-> Chrome
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Chrome

Entities: Chrome, Mozilla, Apple, TechCrunch, Prabhakar Raghavan, Pichai, trial hearing, court testimony, Google antitrust case, decision made with the consumer
in mind, 5 things we learned so far about the Google antitrust case, Samsung, Google, New York Times, iPhones
</item 4>

<item 5>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, Expedia, Jeff Hurst, The Seattle
Times, search engine quality, European antitrust fine, European Union antitrust order, Yahoo, Firefox, Android, antitrust, TechCrunch, DOJ, European Union
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

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, Chrome, Expedia, Yahoo, Firefox,
Android OEMs, antitrust, Microsoft, TechCrunch, Google, Justice Department, iPhones, U.S.
</item 6>

<item 7>
Graph query row:
article_1: 5 things we learned so far about the Google antitrust case
article_1_passages: ['5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)\nGoogle CEO Sundar Pichai testified
Monday in the U.S. government’s antitrust trial against the company. The executive defended Google’s business tactics, including its deal with Apple and other
partners to make Google the default search engine.\n\nThe lawsuit stems from a 2020 antitrust claim over Google’s dominance in the online search market. The
claim is separate from one filed in January by the Department of Justice, along with eight states including New York, California and Colorado, that aims to
“halt Google’s anticompetitive scheme, unwind Google’s monopolistic grip on the market and restore competition to digital advertising.”\n\nThe government has
argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering competitors from accessing
the data they’d need to improve their products.\n\nGoogle argues that it’s just doing business. Everybody wants Google as the default engine because it’s the
best. That doesn’t make its actions illegal, the company says.', '5 things we learned so far about the Google antitrust case (TechCrunch,
2023-10-31T02:30:32+00:00)\nThat doesn’t make its actions illegal, the company says.\n\nMonday’s trial hearing revealed plenty of juicy tidbits, including the
$26.3 billion Google spent making itself the default search engine across platforms in 2021, how Google tried to take it further and have Chrome preinstalled on
iPhones and more.\n\nGoogle paid $26 billion in 2021 to be everyone’s default search engine\n\nWhen Google’s search head Prabhakar Raghavan testified in court
on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple browsers, phones and platforms, from companies including Apple,
Samsung and Mozilla, The Verge reports.\n\nAbout $18 billion of that total amount went directly to Apple, according to a New York Times report published earlier
this month. Google has had its deal with Apple in place since 2003.\n\nWhen questioned on the amount of money Google spends to get first pick of search engines,
Pichai said that the decision was made with the consumer in mind.', '5 things we learned so far about the Google antitrust case (TechCrunch,
2023-10-31T02:30:32+00:00)\nGoogle pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’
devices, said Pichai, according to The Verge.\n\nGoogle understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed
Pichai a 2007 email from a Google product strategy meeting containing data showing that when people changed their browser homepage to Google, they did 15% more
Google searches. When they switched away, their Google searches dropped 27%.\n\n“Nitin argues that focusing on homepage market share is one of the most
effective things we can do to make gains in search market share,” read an email that summarized the meeting and was sent to Pichai, as well as other Google
leaders, according to The Verge.\n\nThe amount that Google spent on homepage market share has been a fixing point in the trial.', '5 things we learned so far
about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)\nThe amount that Google spent on homepage market share has been a fixing point in the
trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified that their search engines would have been far more successful, even competitive with
Google, had they been able to make similar deals with Apple. Microsoft CEO Satya Nadella even said he was willing to spend $15 billion per year to get Bing into
Apple’s default search, per The Information.\n\nGoogle agreed not to promote Chrome to Safari users\n\nAs part of its search deal with Apple, Google agreed not
to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with banners, pop-ups and other annoying means in other Google
apps.\n\nThe agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their
iPhones.\n\nGoogle tried to be preinstalled on iPhones\n\nPichai admitted to attempting to get Tim Cook to preinstall Google on every iOS device back in 2018,
according to The Verge.', '5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)\nHe hoped to make Google and
Apple’s services so connected as to be inseparable.\n\nThe way Pichai pitched it would have been a win-win for both companies. Google gets more people searching
on its platform — not to mention all that juicy data — and Apple would get more revenue, as a result of the lucrative search agreement the two signed.\n\nFor
whatever reason, Cook didn’t take the bait. Apple doesn’t preload third-party software onto its devices, and it wasn’t going to make an exception for
Google.\n\nGoogle’s deleted chat logs\n\nDuring Pichai’s testimony, the DOJ touched on Google’s policy of deleting internal chat messages, despite being subject
to a litigation hold. In February, the DOJ accused Google of systematically destroying chats through its history-off option, which deletes messages every 24
hours unless a user manually changed the setting.\n\nPichai said that he took action against the history-off default for chat in February to comply with the
DOJ’s litigation hold, according to CNBC.', '5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)\nBellshaw pulled
up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a personnel matter and
the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.\n\nA moment of nostalgia for Internet Explorer\n\nCast
your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent Microsoft
an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.\n\nAfter establishing that Google
understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The letter declared that problems
with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”', '5 things we learned so far
about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)\nThese are exactly the arguments that other search engine companies, like DuckDuckGo,
Brave or Microsoft’s Bing, make when they accuse Google of being anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying
Google has become the monopoly it denounced years ago.\n\nWhat does it all mean?\n\nThe case is expected to continue for several weeks, bringing to a head one
of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the 1990s.\n\nIf the judge rules against Google, the outcome could look a lot
like the Microsoft deal, in which the computer company was required to change its behavior and share its APIs with third-party developers. Microsoft was also
banned from making anticompetitive and exclusive deals with computer manufacturers.\n\nGoogle might end up having to turn over all or most of the data it has
collected to other search engines so they can improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does
everyday.', '5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)\nThe DOJ has said that Google gets 16 times more
data than Bing does everyday.\n\nThe Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using
anticompetitive and unfair strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for
third-party apps on its devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls
on Facebook to sell Instagram and WhatsApp.\n\nEnforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most
powerful companies in the world.']
article_2: More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo
article_2_passages: ['More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nMore
from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo\n\nWe’re nearly two months into the Justice Department’s landmark
antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the 1990s — and the revelations just keep
getting juicier.\n\nIn our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how
Google tried to have Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of
the search engine’s most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to
pick with Google.\n\nBefore we go into some of these tidbits…\n\nWhy the Google vs. U.S.', 'More from the US v Google trial: vertical search, pre-installs and
the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nBefore we go into some of these tidbits…\n\nWhy the Google vs. U.S. antitrust case
matters\n\nThe government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus
hindering competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to
change its behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and
computer manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they
can improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.', 'More from the US v
Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nlaw. The Google outcome could also have a
ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to illegally maintain its monopoly
power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether it unfairly favors its own
products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.\n\nThis isn’t Google’s only
antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group. On November 6, Google
went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its Android app store,
Google Play, and its commission structure.\n\nNow, onto the roundup!', 'More from the US v Google trial: vertical search, pre-installs and the case of
Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nNow, onto the roundup!\n\nA window into Google’s most popular search queries\n\nJudge Amit Mehta ruled to
make a list public that provides a glimpse of which search terms make Google the most money. The list of popular search terms ordered by revenue includes 20
terms that were lucrative for the week of September 22, 2018. Information like revenue per search term, how many queries each of those terms got, along with a
separate list of popular search terms ordered by queries (not revenue), were all redacted. The list we can see is as follows:\n\niPhone 8\n\niPhone 8
plus\n\nAuto insurance\n\nCar insurance\n\nCheap flights\n\nCar insurance quotes\n\nDirecTV\n\nOnline
colleges\n\nAT&T\n\nHulu\n\niPhone\n\nUber\n\nSpectrum\n\nComcast\n\nXfinity\n\nInsurance quotes\n\nFree credit report\n\nCheap car
insurance\n\nAARP\n\nLifeLock\n\nThere is, in reality, little surprise here.', 'More from the US v Google trial: vertical search, pre-installs and the case of
Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nWe’ve already established that Google and Apple have a long and mutually beneficial relationship, even
while competing, so it’s not surprising to see three Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the
official release date of the iPhone 8.\n\nMeanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to
how much Google dominates vertical search — that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot
topic in September 2017 and LifeLock was making a big push to win business with people who wanted to buy identity theft protection.\n\nRevenue-shares to
pre-install Google apps on Androids\n\nJamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8.
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.', 'More from the US v Google trial: vertical search,
pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nHe said that the competition between Google and Apple is “as intense as it
gets,” reports Bloomberg.\n\nRosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android
smartphone makers (like Samsung or Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the
default choices, he said.\n\nGoogle also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to
set Google search and Chrome web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The
RSAs also motivate other companies to make or sell more Android devices, he said.', 'More from the US v Google trial: vertical search, pre-installs and the case
of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nThe RSAs also motivate other companies to make or sell more Android devices, he said.\n\nExpedia
complains of too many ads on search, expensive ad payments\n\nOn November 1, Barry Diller, chair of Expedia and IAC, testified about his concerns regarding the
increasing number of ads in search results having an impact on organic listings.\n\n“I must say I’m on the edge of revolt now that Google’s actions are so
punitive, not just for Expedia but also for IAC and all the players that depend upon something of a level playing field,” wrote Diller in a letter to Google
back in 2019, according to Bloomberg.\n\nGoogle CEO Sundar Pichai fired back that Google’s travel listings was one of the most popular experiences the company
built.\n\nExpedia execs also testified about the cost of ads and how increases had no impact on search results.', 'More from the US v Google trial: vertical
search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nOn October 19, Expedia’s former chief operating officer, Jeff Hurst,
told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290 million in 2019. And yet, Expedia’s traffic from Google did not
increase. The implication was that this was due to direct competition from Google itself. Hurst pointed out that Google began sharing its own flight and hotel
data in search results in that period, according to The Seattle Times.\n\nEuropean antitrust fine spurs Google to beef up search engine quality\n\nThe
government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust fine
in 2018, internal documents revealed, according to Bloomberg.\n\nThe EU’s antitrust order forced Google to offer Android phone users a screen with five search
engine options to choose from, according to the DOJ.', 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo
(TechCrunch, 2023-11-13T14:37:44+00:00)\nIn response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and
Germany in 2019 and 2020 with more local content: news, post-game soccer video highlights, information on local television options for streaming, and
pronunciation practice for different language. The aim was to incentivize users to click on Google’s home screen, rather than the competition’s.\n\nThat
revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its products, a
classic consequence in a monopoly.\n\nMozilla digs in on Google’s supremacy in search\n\nInterestingly, on November 1 Mozilla CEO Mitchell Baker provided a
defense of Google’s quality as a search engine, even in “competitive” environments. Specifically, Baker recounted how, Mozilla “failed” when it switched
Firefox’s default search engine from Google to Yahoo.', 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo
(TechCrunch, 2023-11-13T14:37:44+00:00)\nQuick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be
the default search engine on Firefox. At the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.\n\n“I felt strongly that Yahoo was
not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said Yahoo had promised to reduce the
number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.\n\n“The number of users who stayed with Firefox declined
noticeably during the years when Yahoo was the default,” said Baker.\n\nBaker, who was providing a recorded deposition for Google’s defense, noted that
Mozilla’s users apparently wanted and expected Google.\n\nHowever, that is not the full story. Yahoo was already well behind Google in terms of search
technology at that point.', 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch,
2023-11-13T14:37:44+00:00)\nYahoo was already well behind Google in terms of search technology at that point. But also, Firefox was very well behind Chrome,
which ended 2014 with nearly 50% market share and ended 2017 (when Google got prime default search position on Firefox) at nearly 65% of all web browser usage
on desktop: and mobile is even more skewed to Chrome. In other words, the number of Firefox users might have been declining for other reasons, although pushing
the blame on Yahoo certainly works in Google’s favor here!\n\nThe trial continues…']
shared_mentions: ['third-party developers', 'FTC-Facebook case', 'Chrome', 'Sundar Pichai', 'Mozilla', 'antitrust', 'Bing', 'Microsoft', 'Apple', 'WhatsApp',
'monopoly', 'U.S. government', 'DOJ', 'FTC', 'Facebook', 'antitrust law', 'Samsung', 'Google', 'Instagram', 'iPhones', 'iPhone', 'U.S.', 'Bloomberg', 'Amazon']
common_publishers: ['TechCrunch']
direct_article_relationships: []
</item 7>

<item 8>
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
</item 8>

<item 9>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Kent Walker, trial on Google, technology industry, Is Google Search better than the rest? And is that fair?, The Age, Google
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=7d8415bb17a18f4b ===
