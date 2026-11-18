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

This file has 5 items.

=== ITEM id=12f06fed888ce82d ===
Gold sentences and facts:
Sentence 0: Israel stopped allowing deliveries of food, fuel and other supplies to Gaza’s 2.3 million residents on Oct. 10, 2023, and is reportedly preparing for a ground invasion.
  Fact 0.0: Israel stopped allowing deliveries of food to Gaza's 2.3 million residents on Oct. 10, 2023.
  Fact 0.1: Israel stopped allowing deliveries of fuel to Gaza's 2.3 million residents on Oct. 10, 2023.
  Fact 0.2: Israel stopped allowing deliveries of other supplies to Gaza's 2.3 million residents on Oct. 10, 2023.
  Fact 0.3: Israel is reportedly preparing for a ground invasion.
Sentence 1: The understandings that Israel reached with Hamas in 2012 were made "to stop the fighting", Ortal says, and to get life back to normal as quickly as possible.
  Fact 1.0: Israel reached understandings with Hamas in 2012.
  Fact 1.1: The understandings were made to stop the fighting, according to Ortal.
  Fact 1.2: The understandings were made to get life back to normal as quickly as possible, according to Ortal.
Sentence 2: Another user said that no one can carry out such a major attack against Israel without intelligence support.
  Fact 2.0: Another user said that no one can carry out such a major attack against Israel without intelligence support.

Retrieved items:
<item 1>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Ortal points out that between Operation Pillar of Defense in 2012 and Operation Protective Edge in 2014, Israel significantly reduced targeted countermeasures
against terrorist operatives, and none focused on Hamas operatives. According to his analysis, this was because, prior to the Iron Dome coming into use in 2011,
the most dangerous threat posed by Gaza developed in the period preceding Operation Protective Edge. The understandings that Israel reached with Hamas in 2012
were made "to stop the fighting", Ortal says, and to get life back to normal as quickly as possible. This, after air raid sirens sounded in Tel Aviv and
Jerusalem for the first time since the Yom Kippur War, and the education system shut down for more than a week.

Even after Iron Dome, Israel found it difficult to stand up against the significant disruption of daily life on the home front.
</item 1>

<item 2>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Shiloh, which is currently subordinate to the Ministry of Defense, was established following State Comptroller reports about the lack of coordination and
synchronization between the Ministry of Defense and the IDF units; and lead a new border defense concept based on sensors and other technologies for gathering
intelligence and thwarting infiltration; strengthening the Air Force in the form of additional aircraft and armaments; and strengthening the operational end of
the regular front units.

All this, while at the same time closing down armored battalions and consolidating helicopter squadrons. Ground force technological capabilities were also
addressed, such as improved coordination between the infantry and air forces in air-to-ground actions, or identifying terrorists and weapons in an urban
environment. However, there is a problem at the outset, says, Prof. Eviatar Matania of Tel Aviv University School of Political Sciences, Government and
International Affairs, and founder and first director general of the Israel National Cyber Directorate: a large gap between the technology and the officers
expected to implement it as part of the security concept.
</item 2>

<item 3>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Thirdly, as long as the rockets are intercepted in the skies over Israel, and not the skies over Gaza, the Israeli home front is forced to experience sirens,
security rooms and shelters, and disrupted routine living. In other words, even the complete success of the Iron Dome left daily life on the home front in the
hands of the terrorist organizations."

In 2018, Ortal wrote about "the barrier" that, "it would be wrong to assume it will turn out to be more immune than its predecessors. The longer as we delay
cutting the direct connection between the Hamas rockets and those of our other enemies, who make extensive use of this method of operation, and the idea of
attacking our territory, the longer we will be condemned to watch from the sidelines and look at their constant enhancement."

A strategic surprise

In the months leading up to October 7, the former head of the Planning Division, Major General (Res.)
</item 3>

<item 4>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
In an article five years ago, Ortal, currently in active military service an unavailable for interview, wrote, "Israeli restraint as regards proactive thwarting
of offensive capabilities in the Gaza Strip stems from a clear Israeli strategy that strives to reduce the influence of the enemy on the current routine of life
in Israel, even at the price of worsening of the threat in the future." Ortal pointed out that this policy was natural and that "a Western nation cannot lead
its life in an endless state of emergency. But alongside the understanding of the need for a containment policy, it must be recognized that the risks are high,
as evidenced by the steeply worsening intensity of conflicts between Israel and the terrorist organizations in Gaza."

According to Ortal, in recent decades, Hamas and Hezbollah developed ballistic missiles and terror tunnels in response to Israeli policy that sought minimal
friction with the enemy through separation and withdrawal from the security zone in southern Lebanon, and the Gaza Strip.
</item 4>

<item 5>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
On Saturday, October 7, for example, suspicious movement in Gaza convinced the Shin Bet head to send a special team to the southern border, but he was unable to
convince the top IDF echelons to prepare accordingly. To handle those weak signals, Eiland claims, one must use "scenario theory", meaning, taking into account
scenarios that have a low probability of occurring, but with devastating results should they occur, and the simple way to prepare for them.

Eiland explains: "You detect certain noises on Friday night, but conclude that the probability of an attack is low, let's say only 10%. Those who act only on
the probability do not consider the question: if this attack happens - will you be able to live with yourself for eliminating this threat? There’s also a third
consideration, which is ease of effort: what is the effort required of me to prepare for this unlikely but dangerous scenario?
</item 5>

<item 6>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Giora Eiland, held a series of lectures for senior officers on the reasons for the strategic surprise that fell on the IDF in the Yom Kippur War. "Not one of
the officers imagined that 50 years after that mistake, it would repeat itself in exactly the same way," he tells Globes.

The strategic surprise that hit the State of Israel about four weeks ago bears similar characteristics to the Japanese attack on the US naval base at Pearl
Harbor in 1941; to the Argentine surprise attack on Britain's Falkland Islands in 1982; the Al-Qaeda attack on the United States on September 11, 2001; and to
the spread of ISIS throughout the Middle East in 2014.

In fact, Barnea claims, despite the technological progress and sophistication, intelligence officials are questioning whether there really is a trend towards
improvement of early detection and thwarting of strategic threats.
</item 6>

<item 7>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
A comprehensive study conducted on intelligence failures in the national arena, from the fall of the Iron Curtain in 1989 to 2010, showed no improvement in
providing or thwarting warnings about strategic surprises, even if the means of gathering information had improved. Four years after signing the study, former
US President Barack Obama admitted that US intelligence had not correctly assessed the severity of the threat of large parts of Iraq and Syria being occupied by
ISIS.

"No matter how technologically advanced your army or how effective your intelligence gathering system, identifying strategic surprises is a human process that
depends on analyzing what is called, in the intelligence community, 'weak signals,'" Barnea says. "These are pieces of relevant information that have been
conveyed to the organization, but because they are low-key, or because of the information overload around them, the organization fails to identify and interpret
them correctly."
</item 7>

<item 8>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Published by Globes, Israel business news - en.globes.co.il - on November 5, 2023.

© Copyright of Globes Publisher Itonut (1983) Ltd., 2023.
</item 8>

<item 9>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
He said the IDF lost an army base, a general was captured right out of his house. "Invincible "Merkava MK4 MBT's captured, and destroyed...what is wrong with
the Israeli security and defence forces? And (what) MOSSAD did?"

Dr Jaswant Gandhi, a professor of management, said that attack on Israel was a "great Mossad failure". He said India had to learn a lot and keep all
intelligence agencies duly greased and internecine conflict-proof. "Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence
Agencies duly greased and internecine conflict-proof. BHARAT is not immune to #terrorism and internal enemies," he wrote on X.

#Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence Agencies duly greased and internecine conflict proof.
</item 9>

<item 10>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
"The Israelis must be totally losing it right now."

the strategic planning that must have gone into this against one of the most high tech occupation and surveillance systems in the world will be studied by many,
the Israelis must be totally losing it right now https://t.co/oUX2rokEjh — Rania Khalek (@RaniaKhalek) October 7, 2023

Tarik Cyril Amar said: "And the intel, and the special forces training... both areas where Israel mythology will be very upset indeed."

And the intel, and the special forces training... both areas where Israel mythology will be very upset indeed. — Tarik Cyril Amar (@TarikCyrilAmar) October 7,
2023

Mjoachim, another user, said that Israel's defence forces have lost their fame. He said the IDF lost an army base, a general was captured right out of his
house.
</item 10>
=== END ITEM id=12f06fed888ce82d ===

=== ITEM id=a23e2332a5fb9420 ===
Gold sentences and facts:
Sentence 0: The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of seven counts of fraud and conspiracy.
  Fact 0.0: The criminal trial for Sam Bankman-Fried was highly anticipated.
  Fact 0.1: Sam Bankman-Fried is the former CEO of bankrupt crypto exchange FTX.
  Fact 0.2: The criminal trial for Sam Bankman-Fried started Tuesday.
  Fact 0.3: The trial determines whether Sam Bankman-Fried is guilty of seven counts of fraud and conspiracy.
Sentence 1: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 1.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud.
  Fact 1.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth, power and influence.
  Fact 1.2: The defense countered that the FTX founder acted in good faith.
  Fact 1.3: The defense said the FTX founder never meant to commit fraud or steal.
  Fact 1.4: The defense said the FTX founder basically got in over his head.

Retrieved items:
<item 1>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Whatever records the Southern District of New York has for the transactions will be shown.

“What conversations happened between him and his co-conspirators that are now cooperating against him?”

The second part is where all the drama is likely to come, says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law
firm Withers. Prosecutors have to connect those transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it
anyway.

“What was he saying to his parents and his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are
now cooperating against him?”

To further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”

“We had him going out directly to the internet,” LaVigne says.
</item 2>

<item 3>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.
</item 4>

<item 5>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 5>

<item 6>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.
</item 6>

<item 7>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
All the co-conspirators seem oddly younger than their years, as though they got stuck in the “math camp forever” environment Bankman-Fried created for them and
simply failed to grow up.

The prosecution has made a strong case that Bankman-Fried presided over the various forms of fraud that played out at Alameda Research and FTX, orchestrating
some of it and engaging in some of it directly. In the process of making this case, there is a simultaneous drama of ruined lives playing out in the courtroom.

Everyone who loved Bankman-Fried most in the world is now caught up in the case. We have:

Adam Yedidia, Bankman-Fried’s college roommate, who testified with immunity because he was concerned he may have unknowingly worked on code that contributed to
the fraud. “I love you Sam,” he texted at one point. “I’m not going anywhere.” That was before he realized FTX had stolen customer funds — at which point, he
resigned.
</item 7>

<item 8>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Sure, Sam Bankman-Fried’s trial has revealed that FTX was basically a seething mass of scams. But it has also revealed how Bankman-Fried blew up the lives of
the people who are closest to him.

Of course everyone who had money trapped on the FTX exchange when it collapsed — as the result of Bankman-Fried’s trading firm Alameda Research purloining its
customers’ funds — had their lives altered. Even if they are made whole by the bankruptcy proceedings, that’s still money that they should have had access to at
any time they wanted it. That may mean houses that weren’t purchased, medical procedures that were left undone, and life savings lost.

There is a simultaneous drama of ruined lives playing out in the courtroom

Bankman-Fried is charged with seven counts: two counts of wire fraud and five conspiracy charges. As we near the end of the prosecution’s case, I have been
struck by how bizarre it all is.
</item 8>

<item 9>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.
</item 9>

<item 10>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
Come Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of
orchestrating “one of the biggest financial frauds in American history”: the multibillion-dollar collapse of FTX, the crypto exchange he founded and led.

SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy. But, as we wait to hear both his version and the government’s version
of what happened, there will be another tale recounting how things played out: “Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF,
is set to debut on the first day of the trial.

The episode has fired up a lot of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.

60 Minutes got rightly community noted here.
</item 10>
=== END ITEM id=a23e2332a5fb9420 ===

=== ITEM id=c8156a52e7ed8fce ===
Gold sentences and facts:
Sentence 0: When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.
  Fact 0.0: Google's search head Prabhakar Raghavan testified in court on October 28.
  Fact 0.1: Prabhakar Raghavan revealed in court testimony that Google paid $26.3 billion in 2021.
  Fact 0.2: The $26.3 billion was paid to multiple browsers, phones and platforms.
  Fact 0.3: The payments were made to companies including Apple, Samsung and Mozilla.
  Fact 0.4: The Verge reports this.
Sentence 1: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 1.0: The case (the sentence's subject, referred to as 'It') will attempt to argue that Google restrains competition within two separate markets.
  Fact 1.1: One of the two markets is the distribution of apps to Android users.
  Fact 1.2: The other market is the market for payment processing solutions for content inside Android apps.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 2>

<item 3>
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
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 5>

<item 6>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 6>

<item 7>
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
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 9>

<item 10>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 10>
=== END ITEM id=c8156a52e7ed8fce ===

=== ITEM id=6ce409337ccaaddd ===
Gold sentences and facts:
Sentence 0: “Hopefully we can go one better this year,” he added, recalling the 2-0 defeat by Canada in 2022.
  Fact 0.0: The speaker said, 'Hopefully we can go one better this year.'
  Fact 0.1: The speaker recalled the 2-0 defeat by Canada in 2022.
Sentence 1: After a winless campaign in 2022, the Swans – led by ex-Collingwood player Chloe Molloy – have been competitive this season.
  Fact 1.0: The Swans had a winless campaign in 2022.
  Fact 1.1: The Swans are led by ex-Collingwood player Chloe Molloy.
  Fact 1.2: The Swans have been competitive this season.

Retrieved items:
<item 1>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Credit: AP It had followed a superb win from teammate Popyrin, who defied jangling nerves to earn a 7-6 (7-5), 6-2 victory over Otto Virtanen. It meant the
Australians’ crack doubles team of Max Purcell and Matt Ebden weren’t even needed.

It will be Australia’s 49th appearance in the Davis Cup final, as they seek the trophy for the 29th t28-time champions Australia, in a 49th final, seek a first
win in the World Cup of men’s tennis in 20 years. “We’re a very solid squad, a lot of depth in every single position, and if we can get ourselves to a deciding
doubles, we’ve got an unbelievable pairing as well,” de Minaur said. “So we singles guys have just got do our best to try and get a win and let the doubles guys
do the job.
</item 1>

<item 2>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.
</item 2>

<item 3>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The beneficiaries of all this? Lleyton Hewitt’s Australian side, who were happily enjoying a day’s rest at their team hotel further down the Costa del Sol,
watching the drama unfold on TV.

While the Italy tie didn’t finish until 8.15pm after a noon start, Alex de Minaur and co should be fresh for combat in Sunday afternoon’s (Monday AEDT) final as
they seek to lift the Davis Cup for the first time in 20 years. AAP De Minaur, Popyrin power Australia into Davis Cup final Lleyton Hewitt has declared his
close-knit tennis band of brothers will give it “a hell of a shake” to bring back the Davis Cup to Australia after they roared into another final in Malaga.
</item 3>

<item 4>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Instead, in the Davis Cup final in Malaga, Australia will face an Italy team running hot on the heroics of their golden boy Sinner, who beat Djokovic in both
the singles and the doubles in the space of five exhilarating hours in their semi-final on Saturday (Sunday AEDT). Jannik Sinner saved three match points before
going on to beat Novak Djokovic. Credit: Getty Images In one of the great Davis Cup rubbers, Djokovic had held three match points against Sinner and was on the
verge of a completing a relatively workaday 2-0 victory for the Serbs.

But for only the fourth time in his unparalleled career, Djokovic was beaten from match-point up as Sinner roared back to complete his second victory over the
master in just 12 days, winning 6-2, 2-6, 7-5.
</item 4>

<item 5>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
“It was probably the biggest win of my career,” he admitted. “To win a match that means so much to us, it’s an honour and something I’ll never forget. Nothing
compares to this.

“I thought I was ready but when I came out and heard the two sets of fans chanting so hard, but it was nerves that I’ve never experienced before.” Loading It
wasn’t easy. The pressure was on with Thanasi Kokkinakis, the injured teammate he’d replaced, Purcell and Jordan Thompson having all lost opening singles ties
in this year’s competition, and Popyrin struggled too against a big-hitting 22-year-old going for broke. But after repelling a set point in the opening stanza,
Popyrin’s solidity in comparison to his increasingly erratic opponent helped silence a noisy contingent of Finnish fans, who’d been hoping to cheer their
underdogs to a first-ever final.
</item 5>

<item 6>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The 22-year-old Sinner, by now an irresistible force, then re-emerged to team up with Lorenzo Sonego and spray massive forehand winners around liberally as they
outplayed a by now weary-looking Djokovic and Miomir Kecmanovic 6-3, 6-4. Djokovic had three match points in the singles rubber. Credit: Getty Images It was a
crushing end to Djokovic’s hopes of ending his triumphant season, which has featured three grand slam victories and an ATP finals showdown victory over Sinner
just six days ago, with a second Davis Cup title. After telling fans to “shut up” earlier in the week, he again got irritated with the crowd in the doubles,
pretending to be the conductor to the Italian fans’ deafening chorus, but though he may be the greatest player who ever picked up a racquet, doubles looks an
alien game to him.
</item 6>

<item 7>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The second of Sinner’s Djokovic victories came in Italy’s impressive Davis Cup semi-final triumph over Serbia, after saving three consecutive match points on
serve, before besting the world No.1 again in the doubles. World No.4 Sinner, coached by Australia’s Darren Cahill, is the first player to topple Djokovic twice
in the same season since Russia’s Daniil Medvedev in 2019. Italian teammates Lorenzo Sonego and Jannik Sinner will face Australia in the Davis Cup final.
Credit: Getty However, if Popyrin can get the job done in the No.2 singles then Australia will, at minimum, reach the doubles with a live chance, and a good
one, with 2022 Wimbledon champions Max Purcell and Matt Ebden to be unleashed.
</item 7>

<item 8>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Loading Hewitt animatedly embraced Popyrin after his straight-sets victory over 171st-ranked Finn Otto Virtanen, but Australia’s Davis Cup captain would be even
more thrilled if he could repeat the dose against Italy. Australia could also turn to an alternative No.2 singles option, with Jordan Thompson and Purcell
waiting in the wings. “It’s everybody’s dream to play for their country,” Popyrin said. “I looked up to Lleyton, looked up to all the older Aussie legends. To
be able to put on the green and gold, to be able to go out there and play for your country, no matter the format, if it’s this Davis Cup or the old Davis Cup,
you give your all.

“It’s nerves, it’s pressure – you’re playing for everybody back home. It’s something that nothing will ever compare to ... it really was the biggest moment of
my career to get the first live rubber win, to get one foot into the final.
</item 8>

<item 9>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“We’ll continue to keep building the base and keep building the foundations and have it not just strong at year seven, season eight, but strong for generations
to come.”

In a bid to improve the spectacle of the AFLW, the league introduced longer quarters this year and an interchange cap to help increase scoring. The effect,
according to Livingstone, has been positive.

“We’ve increased game time by eight minutes per match, on average, and we’ve had record scores across the rounds,” she said. “Whereas leading into the season we
hadn’t scored more than 100 goals in a round, we’ve regularly been scoring over 100 goals in a round.”

The Swans recorded their highest total last round, in their 14-12-96 to 6-3-39 defeat of the Western Bulldogs, the league’s winless bottom side. Molloy recorded
17 score involvements to set a new AFLW mark.

But Livingstone said the AFL was happy with the level of competition.
</item 9>

<item 10>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
It was something that I’ll never forget.” Loading Australia, who lost last year’s final to Canada, has enjoyed a soft run on paper to this year’s decider,
eliminating the Czech Republic and Finland – with neither country boasting a top-30 singles player – but will have earned their 29th Davis Cup title if they
account for Italy. Canada defeated Italy in the Davis Cup round-robin matches in September, but bowed out to Finland in the knockout stage without Felix
Auger-Aliassime and Denis Shapovalov, who have both battled injury and form. The final is due to start at 2am AEDT on Monday and will be broadcast on 9GEM.

Sinner ends Djoker’s Davis Cup dream Novak Djokovic’s dream of capping his great season by blasting Serbia past Australia in the Davis Cup final was blown up by
Jannik Sinner’s extraordinary double triumph over the world No.1.
</item 10>
=== END ITEM id=6ce409337ccaaddd ===

=== ITEM id=041848e3e6d08b84 ===
Gold sentences and facts:
Sentence 0: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 0.0: The Justice Department is focused on the deals Google makes.
  Fact 0.1: Google makes deals with Apple.
  Fact 0.2: Google makes deals with Samsung.
  Fact 0.3: Google makes deals with Mozilla.
  Fact 0.4: Google makes deals with many others.
  Fact 0.5: The deals are meant to ensure Google is the default search engine on practically every platform.
Sentence 1: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 1.0: The trial will probably bring up the deals later on.
  Fact 1.1: Epic chose to focus more on simply painting Google as the bad guy on day one.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 2>

<item 3>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Cue is one of the highest-profile witnesses in the case so far, in part because the deal between Google and Apple — which makes Google the default search engine
on all Apple devices and pays Apple billions of dollars a year — is central to the US Department of Justice’s case against Google.

Cue had two messages: Apple believes in protecting its users’ privacy, and it also believes in Google. Whether those two statements can be simultaneously true
became the question of the day.

Apple is in court because of something called the Information Services Agreement, or ISA: a deal that makes Google’s search engine the default on Apple’s
products. The ISA has been in place since 2002, but Cue was responsible for negotiating its current iteration with Google CEO Sundar Pichai in 2016. In
testimony today, the Justice Department grilled Cue about the specifics of the deal.

When the two sides renegotiated, Cue said on the stand, Apple wanted a higher percentage of the revenue Google made from Apple users it directed toward the
search engine.
</item 3>

<item 4>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 4>

<item 5>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 6>

<item 7>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=041848e3e6d08b84 ===
