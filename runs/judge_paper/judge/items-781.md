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

=== ITEM id=ac9d964ae7490aa6 ===
Gold sentences and facts:
Sentence 0: Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence.
  Fact 0.0: Google's public documentation says Google uses three categories to rank results that are local to a searcher.
  Fact 0.1: The three categories are relevance, distance and prominence.
Sentence 1: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 1.0: The article's last roundup reported how Google spent $26.3 billion in 2021 making itself the default search engine across platforms.
  Fact 1.1: The article's last roundup reported how Google tried to have Chrome preinstalled on iPhones.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 4>

<item 5>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
</item 5>

<item 6>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 7>

<item 8>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 8>

<item 9>
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
</item 9>

<item 10>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 10>

<item 11>
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
</item 11>

<item 12>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 12>

<item 13>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 13>

<item 14>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Yahoo was already well behind Google in terms of search technology at that point. But also, Firefox was very well behind Chrome, which ended 2014 with nearly
50% market share and ended 2017 (when Google got prime default search position on Firefox) at nearly 65% of all web browser usage on desktop: and mobile is even
more skewed to Chrome. In other words, the number of Firefox users might have been declining for other reasons, although pushing the blame on Yahoo certainly
works in Google’s favor here!

The trial continues…
</item 14>

<item 15>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 15>

<item 16>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.
</item 16>

<item 17>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 17>

<item 18>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 18>

<item 19>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 19>

<item 20>
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
</item 20>

<item 21>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 21>

<item 22>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 22>

<item 23>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
The RSAs also motivate other companies to make or sell more Android devices, he said.

Expedia complains of too many ads on search, expensive ad payments

On November 1, Barry Diller, chair of Expedia and IAC, testified about his concerns regarding the increasing number of ads in search results having an impact on
organic listings.

“I must say I’m on the edge of revolt now that Google’s actions are so punitive, not just for Expedia but also for IAC and all the players that depend upon
something of a level playing field,” wrote Diller in a letter to Google back in 2019, according to Bloomberg.

Google CEO Sundar Pichai fired back that Google’s travel listings was one of the most popular experiences the company built.

Expedia execs also testified about the cost of ads and how increases had no impact on search results.
</item 23>

<item 24>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Looking back on all that had changed around what first attracted people to Google, from the introduction of ads to the efforts to keep users within the universe
of Google products, this seemed to be the last straw.

The second major reason why Google results feel different lately was, of course, SEO — specifically, the obnoxious-kid-refusing-to-behave-in-class kind of SEO.

“SEO that goes against Google’s guidelines, it’s not new,” Ray explained. A decade ago, it used to be called “black hat” SEO, in comparison to the search
engine-approved “white hat” tactics. And Google has, as Sullivan and many SEOs told me, gotten better over time at catching SEOs playing tricks on the
algorithm.
</item 24>

<item 25>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Quick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be the default search engine on Firefox. At
the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.

“I felt strongly that Yahoo was not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said
Yahoo had promised to reduce the number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.

“The number of users who stayed with Firefox declined noticeably during the years when Yahoo was the default,” said Baker.

Baker, who was providing a recorded deposition for Google’s defense, noted that Mozilla’s users apparently wanted and expected Google.

However, that is not the full story. Yahoo was already well behind Google in terms of search technology at that point.
</item 25>

<item 26>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 26>

<item 27>
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
</item 27>

<item 28>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 28>

<item 29>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 29>

<item 30>
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
</item 30>
=== END ITEM id=ac9d964ae7490aa6 ===

=== ITEM id=ca19920018d3a459 ===
Gold sentences and facts:
Sentence 0: In this innovation cycle, product design requires a wider perspective of platforms and interconnections between technologies, creating a strong need for technologists and designers to be in the process together.
  Fact 0.0: In this innovation cycle, product design requires a wider perspective of platforms and interconnections between technologies.
  Fact 0.1: In this innovation cycle, there is a strong need for technologists and designers to be in the process together.
Sentence 1: Stufish partner Ric Lipson (along with artist Es Devlin and producer Brian Eno) collaborated on the U2 project for Sphere.
  Fact 1.0: Ric Lipson is a partner at Stufish.
  Fact 1.1: Ric Lipson collaborated on the U2 project for Sphere.
  Fact 1.2: Es Devlin, an artist, collaborated with Ric Lipson on the U2 project for Sphere.
  Fact 1.3: Brian Eno, a producer, collaborated with Ric Lipson on the U2 project for Sphere.

Retrieved items:
<item 1>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
Hey, folks, and welcome to Week in Review (WiR), TechCrunch’s newsletter covering the past week (or so) in tech industry happenings. This week marked OpenAI’s
first-ever dev conference, where the Microsoft-backed AI startup announced a host of new products. But that was far from the only item of note.

In this edition of WiR, we spotlight Brian’s review of the 16-inch M3 Max MacBook Air and M3 iMac 24-inch; Mozilla betting on a decentralized social networking
future; Ford shuttering a company that was building an app for plumbers, electricians and other trades; and Tim Cook’s thoughts on generative AI. Also on the
agenda is WeWork officially filing for bankruptcy, Bumble getting a new CEO, and the spectacular failure of EV startup Arrival.

It’s a lot to get through, as always — so we won’t delay.
</item 1>

<item 2>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
And if we can solve it, it will do so much good. It really, really will because we will find out we have more in common than not,” he said.
</item 2>

<item 3>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Technologists must “find the grain” of a technology. In this way, they become an intermediary between the abstract nature of design and the pedantic nature of
technology. This philosophical perspective is especially important when a product is in a growth stage or when using emerging technologies.

Regardless of whether your product is growing or in a stable stage or employs established or emerging technologies, integrating technologists into the product
strategy and design process enriches the final outcome. There is a technology perspective extending beyond the operations and mechanics of the code “factory
floor,” which provokes innovation. Sometimes this leads to small, impactful moments of innovation — and sometimes it is a brilliant revolution.
</item 3>

<item 4>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Too often, design is minimized to only the act of making user interfaces pretty. Product design is much deeper and broader in scope than visual design assets.
For example, product design can give direction and focus to a business strategy, user experience strategy, or technology explorations.

The process establishes a guide rail throughout any innovation initiative. The core of product design is decision-making with an astute instinct for making the
best decisions at the most opportune time. Product design helps reduce risk and leads to more effective innovation through quality decision-making.

A progressive role for technologists

Technologists play a strategic role in product innovation and should bring a metaphysical perspective in addition to being punctilious. Our job is to
communicate the essence of a technology and think strategically about the application of technologies to problem spaces. We are most constructive when we
translate the technicals of “how to make X do Y” to “these are the types of products and services realizable with technology X.”
</item 4>

<item 5>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
It won’t always be right, though. But it becomes the epicenter of where the conversation is. And then I’ll think what percentage of our population really wants
to earnestly take the breath and say, let me stop doomscrolling, and let me just wait until people can do the work, synthesize it and then help me react?” he
continued.

“We are also so plugged into all these storylines, whether it’s from traditional media, whether it’s from user-generated — someone on their phone, you know,
posting to a Telegram group. We are overwhelmed and overloaded, and it’s unfortunately appealing to the worst parts of a lot of our instincts,” Ohanian added.

In part, some of the issues around how social media platforms developed had to do with how the companies were originally developed. Ohanian, for example, shared
that Reddit grew out of his running of a forum site in college, when he believed that online community felt as real as offline.
</item 5>

<item 6>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Does Reddit co-founder Alexis Ohanian have regrets over his creation? The entrepreneur and founder of venture fund Seven Seven Six, speaking at an event at the
University of Virginia, espoused a fairly negative view of the current social media landscape and the creator economy — an environment where he says, everyone
has “been so trained and conditioned now to just get the likes and retweets, get the upvotes…I wish I could be more optimistic about it, but we’re seeing it
play out before our eyes.” Still, he ultimately described himself as a “techno-optimist” — an apparent nod to a16z General Partner Marc Andreessen’s recent
diatribe where he claimed technology could solve all our problems.

“Guilty as charged,” Ohanian said, during his interview with Kara Swisher. “I do think that pendulum will swing back. I think even swifter than we realize.”
</item 6>

<item 7>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
The idea, O’Hara said, is to rethink social networking from the ground up.

Ford shutters SaaS app for field work: Ford has shut down VIIZR, a software-as-a-service company that, along with Salesforce, built an app to help tradespeople
like plumbers, locksmiths and electricians to schedule field appointments, send invoices and manage customers, Kirsten reports. VIIZR, which was announced in
December 2021, was a separate company majority owned by Ford, with Salesforce as a minority investor.

Apple bets on generative AI: Apple CEO Tim Cook pushed back against the notion that the company was behind in AI on Apple’s Q4 earnings call with investors, as
he highlighted technology developments that Apple had made recently that “would not be possible without AI.” Cook also said that Apple was working on generative
AI technologies, lending credence to reports suggesting the company is on track to spend $1 billion per year on developing generative AI products.
</item 7>

<item 8>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
That’s at the root of the research we shared from the institute last week. Noted senior research scientist Benjamin Burchfiel:

We’ve seen some big progress with the advent of [large language models], using them to impart this high level of cognitive intelligence into robots. If you have
a robot that picks up a thing, now instead of having to specify an object, you can tell it to pick up the can of Coke. Or you can tell it to pick up the shiny
object, or you can do the same thing and do it in French. That’s really great, but if you want a robot to plug in a USB device or pick up a tissue, those models
just don’t work. They’re really useful, but they don’t solve that part of the problem. We’re focused on filling in that missing piece, and the thing we’re
really excited about now is that we actually have a system and that the fundamentals are correct.
</item 8>

<item 9>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
The specifics of the process and technologies are only important as they pertain to how well they solve the problem, the costs to do so, and the overall end
user experience.

Product innovation is experimental and should not always be expected to yield productive results. It requires a learning curve and patience, as the outcomes are
often ambiguous and unknown. Business leadership can struggle with this perspective because it is indefinite (in terms of results and timelines), and it is
challenging to translate pure technology innovation into value creation. A gap opens between technology and product teams wherein technology teams strain to
articulate the capabilities and value of a technology innovation, leading to unfilled promises and the perception of “technology for the sake of technology” or
quips like “a solution looking for a problem.”

The current hype cycle in AI serves as a great concrete example. For technology or product executives, the challenge is how to do more than check the AI box —
it’s about how to meaningfully incorporate AI into a product. Instead of forcing a technology onto a product, the design process flows to the technologies.
</item 9>

<item 10>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Product design is in a moment of profound change and redefinition as technologies like artificial intelligence (AI) and spatial computing dramatically affect
computing experiences. AI, especially, may have only minor impacts on interface design but will significantly impact the holistic product or ecosystem
experience. Spatial computing, on the other hand, will change human-computer interactions and unmoor our understanding of what a computer is.

In this innovation cycle, product design requires a wider perspective of platforms and interconnections between technologies, creating a strong need for
technologists and designers to be in the process together.

For successful products and businesses, innovation is perpetual. There is a never-ending search to find the next new thing that enhances the user experience,
extends product reach, expands revenue, or all three simultaneously. Product design makes innovation less daunting and increases chances for success because it
is a multidisciplinary process with structures and frameworks to catalyze innovation. Technologists have a role in the process that expands beyond simply
validating the technology or concept, which is their customary responsibility.
</item 10>

<item 11>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
For most technical leads and software developers, this is a mode inversion from our traditional tactical and direct interaction with technology. The context
switch from daily build-and-operate is challenging but is paramount to developing successful and innovative products. We are uniquely positioned to generate
strategic insights translated from dense technical minutiae that drive innovative business cases and product experiences.

A technology innovation must solve a business problem, such as improving operational efficiency, growing existing revenue streams, or generating new ones. The
problem space may be customer-facing (e.g., how can we deliver a new feature?) or internal-facing (e.g., how can we make a process more efficient?). The problem
is the paramount concern. The specific technologies or innovations used to solve the problem are often less critical. We cannot lose perspective of the business
needs; otherwise, the activities become too academic or a paid hobby.

An ordinary domestic analogy is hanging a picture. The hole size, bracket, or tools used to hang the picture are immaterial as long as the picture is on the
wall and straight.
</item 11>

<item 12>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Instead of forcing a technology onto a product, the design process flows to the technologies. In this way, the technology becomes a natural solution.

As experts on a technology or technology stack, we can communicate abstract insights or contribute in more conceptual contexts. Technologists add value to the
product design process by sharing their expertise on a technology’s characteristics. Designers use this information to shape and exploit technologies in the
visual and interaction design process. In this way, technologists inform new interaction models, interface metaphors, and product channels. This involvement
gives confidence and conviction to the promises of design.

Think of digital technologies as a material like paint, stone, or wood. For artisans to create with materials, they must understand the material’s ontology and
phenomenology. An artist must know the differences between oil, acrylic, or watercolor paints because each material has different properties that affect how and
what can be created. Technologists must “find the grain” of a technology.
</item 12>

<item 13>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
It has evolved into working in humanitarian aid, delivering emergency food supplies to people in need in Afghanistan and Turkey.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Another superconductor disappointment: Tim writes that a new, supposedly room-temperature superconducting material isn’t what the scientific community hoped it
would be. With the Nature-published paper detailing the material facing retraction, the odds of researchers discovering a room-temperature superconductor are
looking even longer.

Klarna inches toward an IPO: Mary Ann and Alex write that Swedish fintech Klarna is taking steps toward an eventual IPO. The company has initiated a process for
a legal entity restructuring to set up a holding company in the U.K.
</item 13>

<item 14>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Though he’s glad that access to the technology has broadened, that has come with many complications.

“I don’t like the idea that the internet I’m so nostalgic for was actually an ivory tower,” he said, but on the flip side, he said, we’re all exposed to more
people, in real-time, than we could have ever fathomed. And that can put people at odds with one another and with everyone’s different versions of the truth —
even if their truth is believing in something as far-fetched as the earth being flat, he said, pointing to one example that has a following on Reddit.

Meanwhile, asking platforms like Facebook, Twitter, or Reddit to be in charge of verifying truth and accuracy is a difficult task, he said. Ohanian also called
out TikTok as “an arm for the Chinese government” that’s really good at “feeding us what we want, when we want it” — in other words, not always a fountain of
truth, either.
</item 14>

<item 15>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Before discussing the nontraditional ways technologists participate in product innovation, let’s conceptually examine innovation and product design.

The word innovation has a simple meaning — to introduce something new or a novel method of doing something — yet we frequently inflate it to mean something
magical, grandiose, and world-changing. Innovation can just as easily be mundane and straightforward. The key to innovation is being “new.” The “new” can be
localized to a team, product, process, or business unit. The “novel” can be well-known and established practices not incorporated into your workflow or product.
Sometimes, innovation comes from closing small gaps and isn’t always a grand eureka moment.

Instead of forcing a technology onto a product, the design process flows to the technologies. In this way, the technology becomes a natural solution.

Product design is a process and not a discipline or deliverable. It is (understandably) easy to limit the scope of product design to color choices, content
layout, and aesthetics.
</item 15>

<item 16>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
Certainly most of the R&D budget inside of Toyota is spent figuring out how to make the next car or maybe the car five years from now. But there’s this notion —
and I’m sure Hyundai shares this — that we’re at the once-in-a-century time for transformation in the car industry. Battery electric vehicles are much simpler
to make than cars have been in the past — no engine, no transmission, etc. So, we’re going to have to compete in a much fiercer way in cars. But also, can we
use the skills, the dreams and the hopes of the company to go beyond cars? TRI’s job is to actually think about [what’s] next. What’s next after cars, as well
as some fancy stuff on cars.”

For TRI, much of the “what’s next” focuses on supporting aging populations. The organization invests a good portion of its resources to building out technology
designed to help older people live more independently. That’s at the root of the research we shared from the institute last week.
</item 16>

<item 17>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Not Sonos fast there: Audio company Sonos scored a big $32.5 million win against Google a while back. Now it transpires that the company’s patents were deemed
unenforceable and invalid. Whoops. A federal judge threw out the $32.5 million win this week.

When it clicks, it really clicks: We took a deep look at Pixel 8, and our team discovered that it delivers solid performance and design upgrades. The camera got
a particularly enthusiastic shout-out, with Darrell declaring that Google’s Pixel 8 Pro camera is the new mobile photography champ, and Brian waxing lyrical
about the phone’s picture-snapper in “The camera’s still the thing.”

Who needs computers anyway?: It seems like all our mobile devices may be starting to cannibalize sales of personal computers — Ron reports that PC shipments
decline slows in Q3 2023 and that Apple’s sales plunge over 23%.

Let’s get together

Reddit’s API-powered chaos continues, but it appears that things are starting to resolve a little.
</item 17>

<item 18>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
I think the leadership at Hyundai — the chairman visited me last week, and we’re in close touch with him and the people who work with him — think that getting
into the 22nd century is an important thing to do and we’re getting started on it now.”

Pratt adds, “There’s actually this alignment that’s happening between cars and robotics, as well. Rod Brooks was the one who a few years ago said that modern
cars are eldercare robots. I think that’s really true. You think about the amount of computers in them, the amount of software that’s in them. TRI has different
divisions inside of it, one of which works on very advanced things in cars that is completely overlapping with some of the software and some of the concepts
that we’re using in the robotics space.”

For its part, the Boston Dynamics AI Institute lists three key pillars of its research: intelligence, dexterity and mobility. That’s effectively making robots
that are smarter, better at mobile manipulation and move more dynamically.
</item 18>

<item 19>
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
</item 19>

<item 20>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
He found the iMac to be lacking and not necessarily worth the upgrade from the 2021 model, excepting the M3 chip, which brings “impressive” performance gains
over the already-powerful M1. As for the M3 Max MacBook Pro, Brian reports that, at $2,500 (plus some pricey add-ons), it successfully splits the difference
between the Mac Studio and MacBook Air.

Mozilla bets on a decentralized future: Sarah spoke with Mozilla senior director of content Carolyn O’Hara, who outlined Mozilla’s strategy where it concerns
the “fediverse” — a collection of decentralized social networking applications, like Mastodon, that communicate with one another over the ActivityPub protocol.
The idea, O’Hara said, is to rethink social networking from the ground up.
</item 20>
=== END ITEM id=ca19920018d3a459 ===

=== ITEM id=85448776fddfbe41 ===
Gold sentences and facts:
Sentence 0: The Cowboys opened Week 13 by rallying to beat the Seahawks on Thursday night, keeping the heat on the losing Eagles ahead of them before hosting that team in Dallas in Week 14.
  Fact 0.0: The Cowboys opened Week 13 by rallying to beat the Seahawks on Thursday night.
  Fact 0.1: The Cowboys' win kept the heat on the Eagles.
  Fact 0.2: The Eagles were a losing team.
  Fact 0.3: The Eagles were ahead of the Cowboys.
  Fact 0.4: The Cowboys will host the Eagles in Dallas in Week 14.
Sentence 1: Fueled by a stout defensive showing, a gritty run game and a bit of fairy dust from the likes of St. Brown, Sam LaPorta and Josh Reynolds, the Lions cruised to a 34-20 win over their hated NFC North rivals, Green Bay.
  Fact 1.0: The Lions' win was fueled by a stout defensive showing.
  Fact 1.1: The Lions' win was fueled by a gritty run game.
  Fact 1.2: The Lions' win was fueled by a bit of fairy dust from St. Brown.
  Fact 1.3: The Lions' win was fueled by a bit of fairy dust from Sam LaPorta.
  Fact 1.4: The Lions' win was fueled by a bit of fairy dust from Josh Reynolds.
  Fact 1.5: The Lions cruised to a 34-20 win.
  Fact 1.6: The Lions beat Green Bay 34-20.
  Fact 1.7: Green Bay is the Lions' hated NFC North rival.

Retrieved items:
<item 1>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
It's a three-touchdown lead for Seattle now with just over five minutes to play.

Seahawks 24, Giants 3

10:59 p.m.: INTERCEPTION! The nightmarish game for Daniel Jones continues as he commits his third turnover of the game, a pass intercepted by Seahawks safety
Quandre Diggs.

After being sacked nine times and committing three turnovers, Jones may not have much more time on the field tonight.

10:56 p.m.: New York forces a three-and-out to get the ball back and on the first offensive play, Daniel Jones is sacked again.

Jones has now been sacked nine times in the game. It's a rough showing considering Seattle entered the week with just five sacks on the season.

10:48 p.m.: …And ANOTHER sack.

Daniel Jones has now been sacked eight times in the game and New York will be forced to punt as it faces fourth-and-31.

Fans are starting to leave the Meadowlands as it's been all Seahawks tonight.
</item 1>

<item 2>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
New York has been dominated at home twice this season now, as this performance follows a 40-0 blowout in Week 1 on "Sunday Night Football." Daniel Jones' career
mark in primetime falls to 1-13 and 0-3 this season.

11:09 p.m.: TURNOVER ON DOWNS! Daniel Jones is sacked for an 11th time as Seattle takes back over at the New York 35-yard line.

Geno Smith's night appears to be done as Drew Lock is back in for the Seahawks.

11:06 p.m: Daniel Jones is back under center for New York, which will look to drive and score to at least lessen the damage on the scoreboard.

11:04 p.m.: FIELD GOAL! The turnover leads to three points for the Seahawks, who get a 34-yard field goal from Jason Myers. It's a three-touchdown lead for
Seattle now with just over five minutes to play.
</item 2>

<item 3>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Amon-Ra St. Brown was in the midst of a tempest, shrouded in a sea of green with few options around him. The Lions WR had done the hard part, stretching beyond
an exasperated Packers defense to trot into the end zone.

But he had a choice to make: rollick in the joy of his sapphire-clad teammates. Or take a leap of faith into the Lambeau Field abyss.

He opted for the latter. St. Brown hurled himself into the green lagoon. Then, he smiled.

St. Brown's impersonation of the Lambeau Leap was one of many high points in a memorable night for Detroit. Fueled by a stout defensive showing, a gritty run
game and a bit of fairy dust from the likes of St. Brown, Sam LaPorta and Josh Reynolds, the Lions cruised to a 34-20 win over their hated NFC North rivals,
Green Bay.
</item 3>

<item 4>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Their playoff hopes look much better after their stunning comeback vs. the Ravens.

9. Jacksonville Jaguars 6-3 (4)

The Jaguars got an NFC wake-up call against the 49ers, who were much more prepared after a Week 9 bye than they were. Jacksonville needs to forget that debacle
and get back on track, stat. The team gets a division double vs. Tennessee and Houston before facing Cincinnati.

10. Dallas Cowboys 6-3 (11)

The Cowboys took advantage of their "no contest" vs. the Giants at home with Dak Prescott continuing to ball as a passer and runner. They need to keep the
defense and running game success, too, in bigger games going forward.

MORE: Inside the Cowboys historic offensive explosion vs. Giants

11.
</item 4>

<item 5>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
— The Purdy-to-Aiyuk connection continues to get stronger. Aiyuk goes for 23 yards to give the 49ers some breathing room after opening their drive with a false
start.

49ers 21, Cowboys 10

10:17 p.m. FIELD GOAL — The 49ers' defense tightens the screws and limits Dallas to a field goal. Brandon Aubrey drills it from 50 yards out, and it's down to a
21-10 lead for San Francisco.

10:13 p.m. — Prescott hits CeeDee Lamb for a big 29-yard gain. That quickly takes the Cowboys into 49ers territory to start the second half.

End of first half: 49ers 21, Cowboys 7

9:58 p.m. — San Francisco's defense is all over the Cowboys yet again. It's Dre Greenlaw's turn to sack Prescott, and Dallas is waving the white flag on the
half.
</item 5>

<item 6>
NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season (Sporting News, 2023-12-03T21:49:04+00:00)
7. Green Bay Packers (6-6), third place, NFC North. The Packers moved into the final playoff spot thanks to having the second-best conference record after the
Vikings, to whom they already lost but face again in Minnesota in Week 17. The Packers also hold key wins over the Rams and Saints in the wild-card race.

Who's in the hunt?

8. Los Angeles Rams (6-6)

9. Seattle Seahawks (6-6)

10. Tampa Bay Buccaneers (5-7)

11. New Orleans Saints (5-7)

12. New York Giants (4-8)

13. Chicago Bears (4-8)

14. Washington Commanders (4-9)

The Rams and Seahawks are the best wild-card bets here now, while the Saints and Buccaneers will still have their sights on making the playoffs via still
winning the weak South over the Falcons.
</item 6>

<item 7>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
For a Cowboys team that has embarrassed a few opponents this year, they might be on the bad end of an ugly game here.

49ers 28, Cowboys 10

10:27 p.m. TOUCHDOWN — Not one, not two, THREE touchdowns for George Kittle. The 49ers have everything working offensively tonight, and any Kittle fantasy
owners are thrilled.

10:25 p.m. — Aiyuk's loss is Deebo Samuel's gain. Samuel goes 43 yards on the very next play, which tells you how automatic the 49ers' offense is right now. San
Francisco is in the red zone looking to make this a three-score game.

10:23 p.m. — An offensive holding call wipes out a 40-yard gain from Aiyuk. Wow. That's a momentum swing. Spencer Burford is the culprit.

10:21 p.m. — The Purdy-to-Aiyuk connection continues to get stronger.
</item 7>

<item 8>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
Adams was seen slamming his helmet down on the sideline in the fourth quarter, and it's tough to blame the star receiver after an 11-yard performance that could
have been exponentially better with better quarterback play.

The story of the night was Lions rookie Jahmyr Gibbs, who finally broke through after entering the season with sky-high expectations. Tasked with leading the
backfield in David Montgomery's absence, Gibbs exploded for 189 scrimmage yards and a touchdown in the win. It was the first 100-yard rushing performance of his
career.

Meanwhile, Amon-Ra St. Brown notched his third consecutive 100-yard game despite entering Monday as questionable with an illness.

MORE: Why Davante Adams is unlikely to be traded before deadline

The Lions enter their bye in sole possession of the No. 2 seed in the NFC, while the Raiders drop their second consecutive game against an NFC North opponent
and have what might be a must-win game against the Giants in Week 9 if they want any chance to linger in the AFC playoff race.
</item 8>

<item 9>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
It was all Seahawks on “Monday Night Football” as the team turned in a dominant defensive effort en route to a 24-3 win over the Giants to wrap up Week 4 of the
NFL season.

In addition to sacking Daniel Jones 11 times, Seattle’s defense forced New York’s QB to commit three turnovers — a fumble and two interceptions, none more
notable than rookie Devon Witherspoon's 97-yard pick-6 that turned the game on its ear.

With the win, the Seahawks move to 3-1 on the season, including a perfect 2-0 record away from home. The Giants drop to 1-3 on the season and have been
outscored 64-3 in two primetime home losses.

The Sporting News tracked score updates and highlights from throughout Monday night's meeting between New York and Seattle.

Seahawks vs.
</item 9>

<item 10>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
the Bills, vs. 49ers and at the Cowboys in the next four. They still look like the conference team to beat as they have proved they still win in many ways
around Jalen Hurts.

2. Kansas City Chiefs 7-2 (3)

The Chiefs had a pretty good week off given they jumped back into the AFC's top seed and home-field advantage spot with the Ravens losing. They already are
rolling defensively and expect them to figure out a lot more around Patrick Mahomes after the key Andy Reid bye.

3. Detroit Lions 7-2 (5)

The Lions are becoming like the Eagles, and that's what makes them the second-best team in the NFC. They can win shootouts with Jared Goff, Amon-Ra St. Brown
and their passing game. They also can dominate with the running game. They can win with offense or defense. Dan Campbell is coach of the year.

4.
</item 10>

<item 11>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
Jerry Jones said earlier in the week that Sunday night's game would show him how the Cowboys stack up against the NFL's best.

The answer? Not very well.

Everything that could go wrong for Dallas on Sunday did go wrong, in a game that confirmed a 49ers team that ended the Cowboys' season in each of the last two
years remains a level above its NFC counterpart.

The 49ers carved up Dallas' defense on the opening drive, and that was a sign of what was to come. Brock Purdy threw four touchdowns in the 42-10 rout,
including three to George Kittle. Christian McCaffrey scored a touchdown for a 14th consecutive game as all of San Francisco's offensive weapons got involved.

MORE: Comparing stats for NFC stars Nick Bosa, Micah Parsons

Defensively, the 49ers were relentless. Nobody had a larger impact than Fred Warner, who finished the night with an interception, sack and forced fumble in a
swarming performance.
</item 11>

<item 12>
NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season (Sporting News, 2023-12-03T21:49:04+00:00)
They got to .500 and increased their lead to a full game by beating the Jets in Week 13.

5. Dallas Cowboys (9-3), second place, NFC East. The Cowboys opened Week 13 by rallying to beat the Seahawks on Thursday night, keeping the heat on the losing
Eagles ahead of them before hosting that team in Dallas in Week 14. They can finish No. 1, No, 2 or No. 3 if they win the division and are locked into No. 5
again otherwise.

6. Minnesota Vikings (6-6), second place, NFC North. The Vikings jumped up a spot during their bye with the Seahawks losing on Thursday night to match their
record on Thursday night. They have the tiebreaker over the three other .500 NFC teams based on conference record, but right behind them is the team in third
place in the division.

7. Green Bay Packers (6-6), third place, NFC North.
</item 12>

<item 13>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
The 49ers knocked the Cowboys out of the playoffs in both of the last two seasons, so a win for Dallas on Sunday could be helpful just from a psychological
standpoint.
</item 13>

<item 14>
Prisco's Week 12 NFL picks: Jaguars beat Texans in AFC South battle, Saints get key divisional win vs. Falcons (CBSSports.com, 2023-11-23T21:30:38+00:00)
Thursday, 12:30 p.m. ET (Fox, fubo)

This is a big game in the division for both teams, a survival test for Green Bay and a big one for the Lions if they want to try and get home-field advantage in
the conference. The Lions have had defensive issues as of late and Jordan Love has played well the past two weeks. That will keep the Packers in this game, but
the Lions win it.

Pick: Lions 31, Packers 27 (Result: Packers 29, Lions 22)

Thursday, 4:30 p.m. ET (CBS, Paramount+)

The Commanders are coming off a dud last week against the Giants, while Dallas is flying high. The Cowboys are playing as well as any team in the league.
Washington's offensive line is an issue, which will lead to the Cowboys having a field day with their defensive front. Dak Prescott plays well again. Cowboys
roll.
</item 14>

<item 15>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
The Lions just needed to get themselves back in the win column after a blowout loss in Baltimore, and they did just that in front of their home fans on Monday
night.

Detroit rolled to a 26-14 victory over the Raiders in a game that felt much more one-sided than the score indicates.

The Lions thoroughly outplayed the Raiders, out-gaining Las Vegas by 329 yards, but critical mistakes by Detroit left points on the board. The offense struggled
to turn red zone opportunities into touchdowns in the first half, and two turnovers deep in Raiders territory -- including a pick-six by Jared Goff -- kept Las
Vegas in the game in the second half.

MORE: What to know about Lions' retro 'Honolulu blue' helmets

Nonetheless, the Lions' defense dominated. The Raiders were abysmal on the offensive side of the ball, outside of a second-quarter touchdown drive led by Josh
Jacobs. Jimmy Garoppolo completed just 10 passes, badly missing a wide-open Davante Adams on multiple occasions.
</item 15>

<item 16>
Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-12-03T23:00:04+00:00)
They're back in the postseason race, all on the back of their lithe young hurler.

Miss any of the game? The Sporting News tracked live updates from Sunday's clash under the lights at Lambeau Field below.

Chiefs vs. Packers final score

Q1 Q2 Q3 Q4 Final Chiefs 3 3 6 7 19 Packers 7 7 7 6 27

Chiefs vs. Packers results, highlights from 'Sunday Night Football'

(All times Eastern)

11:28 p.m.: Game. Set. Match. Mahomes' Hail Mary falls harmlessly to the turf, giving the Packers a much-needed win. What a game!

11:25 p.m.: Proper make-up call by the referees there, as Valdez-Scantling was almost certainly impeded illegally while streaking down the field. Should've been
a pass interference. It wasn't, though.
</item 16>

<item 17>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
WEEK 13 DFS:

DK lineup | FD lineup | Best stacks | Best values

Cowboys: Who to start, sit in fantasy football on TNF

Start: QB Dak Prescott

Guess the only two quarterbacks who have been better than Rayne Dakota in fantasy this season? Yep, after his sizzling seven-game stretch, Prescott has ascended
behind just Josh Allen and Jalen Hurts. The Seahawks haven't given up many big games to QBs, but you need to keep rolling with Dak on his heater, especially at
home. One can't be surprised, however, if it's more of a 250-and-2 kind of night.

Start: RB Tony Pollard | Sit: RB Rico Dowdle

Pollard and the Cowboys' defense is a reason why Prescott may not need to go nuts passing. The Seahawks have now joined the bottom five teams in giving up
fantasy points to running backs.
</item 17>

<item 18>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
Jourdan Lewis. JSN can mirror some of the production Curtis Samuel gave the Commanders last week in Dallas, but forget 100 yards. Around 50 yards seems like the
ceiling with a big boost from garbage time.

Sit: TEs Noah Fant and Will Dissly

Noah Gray is a better fantasy player than Noah Fant this season. Will Dissly is barely a better fantasy player than Will Mallory this season.

WEEK 13 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Sit: Seahawks D/ST

The Seahawks are getting better defensively against the pass, but that's not enough to trust them here, as they're getting gashed on the ground, too.

Sit: K Jason Myers

This is actually not a good spot for one of the most reliable fantasy kickers, given the Cowboys don't give up many field-goal attempts.
</item 18>

<item 19>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
11:11 p.m.: FIELD GOAL! Carlson's try from distance makes the deficit two possessions.

11:08 p.m.: Not so fast. ... A quick three-and-out gives Green Bay a glimmer. That glimmer grows even brighter after an impressive return by Keisean Nixon.
Let's see what they can do with 3:30 left.

11:04 p.m.: INTERCEPTION! That's wraps. Love misses his target, instead finding the mitts of Jacobs for his second pick of the day. Strong showing by this
Detroit defense. And that should put things to bed.

10:54 p.m.: TOUCHDOWN! Walker's penalty proves vital as Montgomery finds pay dirt from a couple yards out on fourth down. Gutsy call by Campbell. And one that
pays off in spades for Detroit, which extends its lead to 17.

Lions 34, Packers 17

10:50 p.m.
</item 19>

<item 20>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
9:15 p.m.: FIELD GOAL! The good times keep on rolling for Detroit, which adds to its lead after another standout drive spearheaded by Goff and Montgomery. The
Lions are rolling early.

Lions 17, Packers 3

End of first quarter: Lions 14, Pacers 3

8:51 p.m.: Impressive play-calling for the Lions, who continue to carve their way up the field.

8:50 p.m.: Love has been unable to find a rhythm in his first few possessions. Another gives Detroit another chance to make the Lions' life a living hell.

8:43 p.m.: TOUCHDOWN! Montgomery goes into his right tackle before finding the end zone. Detroit's office os cooking with gan.

Lions 14, Packers 3

8:42 p.m.: Decent scamper by Goff brings Detroit right to the cusp of pay first. Lions are closed to adding a second score.
</item 20>

<item 21>
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
</item 21>

<item 22>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
Not allowing even a field goal is a huge win for Dallas.

8:56 p.m. — The 49ers are excelling on third down early in this game, and Purdy flips it to Kyle Juszczyk to keep the Niners' drive alive on 3rd & 1.

8:54 p.m. — Now Kearse lines up offsides... That extends the drive for the 49ers, who aren't moving as freely as they were on that opening drive.

8:50 p.m. — Jayron Kearse comes through with the Cowboys' first sack of the night, but Deebo Samuel makes up 11 of the 12 yards San Francisco needs on the next
play.

8:46 p.m. — There's the first catch of the night for Lamb, but he doesn't get the first down and Dak Prescott is sacked by Nick Bosa to quickly end the drive.
This 49ers defense has picked up right where it left off against the Cowboys in January.
</item 22>

<item 23>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Week 10 in the NFL had some of the league's best teams not playing. But that didn't stop the excitement, with five teams winning on walk-off field goals and
more games coming right down to the wire.

With that, there were some surprising results and expected domination. Some favorites dug deep to win. Some underdogs turned in shocking upsets.

Putting the latest wins and losses into a blender, here's Sporting News' updated look at how all teams stack up against each other, 1-32 going into Week 11:

NFL WEEK 11 PICKS: Straight up | Against the spread

NFL power rankings Week 11

1. Philadelphia Eagles 8-1 (previous week: 1)

The Eagles enjoyed a bye trying to get a little healthier for the more difficult stretch run in a top-heavy NFC as they have tough games at the Chiefs (the
Super Bowl 57 rematch), vs. the Bills, vs.
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
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Easy peasy!

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Both teams had a week of rest, so don't treat this like a typical Thursday game. But do expect the Cowboys defense to keep the pressure on Geno Smith and a
Seattle offense that's struggled to gain consistency over the last month. Smith actually did OK completing passes last week when pressured (60% completion rate)
but not at an efficient level (5.4 yards per attempt). Just two QBs have exceeded 250 yards and four QBs have thrown multiple touchdowns against Dallas this
year; I don't like Smith's chances to do either, which hurts any upside case for any Seahawks receiver. By the way, Curtis Samuel was the first receiver to get
100 yards on the Cowboys this year, and it took him 12 targets and 9 catches to get there.
</item 25>

<item 26>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
MORE: Inside the Cowboys historic offensive explosion vs. Giants

11. Seattle Seahawks 6-3 (13)

The Seahawks needed every ounce of their offense around Geno Smith to rally to beat the Commanders, but the get-well performance worked well for him, Kenneth
Walker and the wide receivers. The defense could be more consistent.

12. Minnesota Vikings 6-4 (14)

The new man on the Vikings, Josh Dobbs, keeps producing the same results they were enjoying offensively with Kirk Cousins, thanks to more help from the running
game for Kevin O'Connell and the defense overall under Brian Flores. Their 0-3 and 1-4 starts are distant memories as the Vikings look for a sixth straight win
in Week 11 vs. Denver.

13. Houston Texans 5-4 (16)

The Texans are looking great with C.J. Stroud leading their offense and now they've found some running game, too.
</item 26>

<item 27>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
21. New York Jets 4-5 (18)

Aaron Rodgers can't come back soon enough, and going for a December miracle might be too late after two straight key losses to the Chargers and Raiders. Zach
Wilson will always cap the potential of a talented offense that keeps making mistakes. The Bills and Dolphins in back-to-back division games next might be their
season.

22. Denver Broncos 4-5 (24)

The Broncos got another big win against a AFC power, sandwiching their Week 9 bye with upsets over the Chiefs and Bills. The offense has hit a balanced grooved
with Russell Wilson and the defense is making a lot more plays, too.

23. Washington Commanders 4-5 (21)

The Commanders should be pleased about Sam Howell and the offense showing more fight with additional weapons and the running backs look better all-around again.
But the reeling defense and pass protection continue to be un-winnable assets.

24.
</item 27>

<item 28>
Monday Night Football DraftKings Picks: NFL DFS lineup advice for Week 15 Eagles-Seahawks Showdown tournaments (Sporting News, 2023-12-18T22:47:00+00:00)
The Seahawks are just about hanging on by a thread. At 6-7, the team is on the outside looking in when it comes to the playoff picture. A win on Monday night
against the suddenly reeling Eagles could be a big boost in their chances going forward -- a matchup that is actually a pretty good one after looking beneath
the surface. The Eagles come in as losers of two straight, losing in blowout fashion both times no less. Suddenly the sky is falling as the Cowboys have taken
over the NFC East division lead, complicating matters as the regular season winds down. The sense of urgency for both sides means we are looking at two teams
with plenty to lose, setting up for a fun Monday night under the lights at Lumen Field.
</item 28>

<item 29>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
Atlanta entered tied atop the division with New Orleans, who lost to Detroit.

But this game appealed only to fans of sloppy play, punting and penalty flags. The Jets (4-8) got an early safety but couldn’t get anything going – again – on
offense in losing their fifth straight game. New York even pulled quarterback Tim Boyle and replaced him with Trevor Siemian in the fourth quarter to try to
provide a spark, but it was more of the same misery in what appeared a half-filled MetLife Stadium.

Cleveland Browns 19–36 Los Angeles Rams

Matthew Stafford threw three touchdown passes, Puka Nacua became the first Rams rookie wide receiver to reach 1,000 yards in a season, and Los Angeles extended
their winning streak to three games. The Rams have won three straight for the first time since winning the Super Bowl after the 2021 season. At 6-6, they are
back in the playoff race after dropping three straight before their bye week.
</item 29>

<item 30>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
Carr was hurt on a penalized hit by Bruce Irvin. New Orleans fell to 5-7, behind first-place Atlanta in the NFC South.

Kansas City Chiefs 19-27 Green Bay Packers

The Green Bay Packers picked up their fourth win in five games as they beat the defending Super Bowl champions. The 6-6 Packers are now in the NFC wildcard
places after Sunday night’s win.

Denver Broncos 17-22 Houston Texans

Nico Collins had a career-high 191 yards receiving and a fourth-quarter touchdown, and Jimmie Ward intercepted Denver’s Russell Wilson in the end zone with nine
seconds left to left to help the Houston (7-5) Texans hold on for a win over the Broncos. Ward leapt in front of intended receiver Lucas Krull to secure the
fourth victory in five games for Houston and snap a five-game winning streak for Denver (6-6). Wilson was intercepted a season-high three times, all of them
after halftime.
</item 30>

<item 31>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They also seemed invigorated for Jonathan Gannon on defense and special teams. Look for the Cardinals to finish strong and play a whole lot of spoiler.

30. New England Patriots 2-8 (28)

The Patriots tried to win a game by hiding Mac Jones and their quarterbacks in general. They almost pulled it off with the running game dominating and
short-handed defense making a lot of plays, but they didn't see the results in Germany ahead of a tough bye for Bill Belichick.

31. New York Giants 2-8 (30)

The Giants can just flush away this season without Daniel Jones and Tyrod Taylor. The injuries go to their offense early and remained relentless, and now their
defense is falling apart, too. The question is, will Brian Daboll get a massive mulligan after his rookie playoff leadership?

32.
</item 31>

<item 32>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Lions 34, Packers 17

10:50 p.m.: Quay Walker's leaps over a blocker on Riley Patterson's field goal attempt, prompting a unsportsmanlike penalty. That will put Detroit on the cusp
of another score, pending a Packers defensive rally.

10:43 p.m.: Lions beginning to salt the game away with strong running through the tackles. Montgomery and Gibbs creeping through the crevices to push Detroit
down the field. And an impressive toss from Goff into the outstretched hands of Reynolds, as well. The Lions are on the doorstep of the red zone.

10:36 p.m.: TOUCHDOWN! Love scampers into the end zone from short distance to cut that Detroit lead down even more. He got held short of the end zone on the
ensuing two-point effort. But Green Bay is firmly back in it.
</item 32>

<item 33>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
When the Seahawks and Cowboys meet in Dallas on "Thursday Night Football" to open NFL Week 13 (8:15 p.m. ET, Amazon Prime Video), it will be a prime-time
matchup featuring two potential playoff teams and plenty of key fantasy football start 'em, sit 'em question marks.

With a game total of 47.5 and the Cowboys favored by three field goals, it may be another rout for the home team against a weaker opponent. Dallas has some
red-hot fantasy assets, while Seattle's struggling offensive skill players might be best suited for fantasy benches.
</item 33>

<item 34>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
NFL WEEK 4 PICKS: Straight up | Against the spread | Projection model

The Sporting News tracked live updates from Thursday's clash between the Lions and the Packers. Scroll below to check out highlights and live results as two of
the NFC's oldest sides faced off for NFC North glory.

Lions vs. Packers score

1 2 3 4 F Lions 14 13 0 7 34 Packers 3 0 8 9 20

Lions vs. Packers live updates, highlights from Thursday Night Football

(All times Eastern)

End of fourth quarter: Lions 34, Packers 20

11:19 p.m.: Lions complete yet another fourth down, officially putting the game on ice.

11:12 p.m.: Onside kick spills out of bounds, and it's looking like that might be that. Packers have one timeout remaining but Montgomery's fiery run through
the tackles should make that obsolete.

11:11 p.m.: FIELD GOAL!
</item 34>

<item 35>
Prisco's Week 12 NFL picks: Jaguars beat Texans in AFC South battle, Saints get key divisional win vs. Falcons (CBSSports.com, 2023-11-23T21:30:38+00:00)
Dak Prescott plays well again. Cowboys roll.

Pick: Cowboys 32, Commanders 16 (Result: Cowboys 45-10)

Thursday, 8:20 p.m. ET (NBC, fubo)

This is a huge division game for two teams that have had a good rivalry over the past decade. The Seahawks are coming off a bad loss to the Rams, while the
49ers are coming off two dominant victories. Geno Smith is banged up, so his health is key. The 49ers have the look of a Super team again. They show it here.

Pick: 49ers 28, Seahawks 19 (Result: 49ers 31-13)

Friday, 3 p.m. ET (Amazon Prime Video)

This Friday game the day after Thanksgiving isn't what it was expected to be with the Jets being an offensive disaster.
</item 35>

<item 36>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
Fans are starting to leave the Meadowlands as it's been all Seahawks tonight.

10:45 p.m.: Daniel Jones has now been sacked seven times in the game.

Bobby Wagner gets Jones again, setting up second-and-28 for the Giants, who were already at a disadvantage thanks to a penalty on first down.

End of third quarter: Seahawks 21, Giants 3

10:33 p.m.: PICK SIX! In the worst possible outcome for the Giants, Devon Witherspoon sits on an underneath route and intercepts Daniel Jones, returning the
pick 97 yards for a touchdown. The PAT is good!

What could have been a one-possession game is now a three-score game. And Daniel Jones now has two turnovers in this one.

Seahawks 21, Giants 3

10:30 p.m.: The Giants just pulled off one of their biggest plays of the night.
</item 36>

<item 37>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
End of first quarter: 49ers 7, Cowboys 0

9:04 p.m. — Confirmed: 49ers ball. San Francisco has an instant opportunity here.

9:01 p.m. FUMBLE — Oh no! Pollard gets the ball knocked out deep in the Cowboys' own territory, and it looks like it was recovered by the 49ers. Officials are
checking to make sure nobody was out of bounds, but that ball was very much out of Pollard's control. The 49ers could be set up in the red zone.

8:59 p.m. — The drive completely falls apart for the 49ers. A delay of game penalty sets San Francisco back, and Christian McCaffrey is dropped for a five-yard
loss on 3rd & 15. Not allowing even a field goal is a huge win for Dallas.

8:56 p.m.
</item 37>

<item 38>
NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 (The Guardian, 2023-12-03T22:11:26+00:00)
The 49ers spent an offseason chirping at the Birds – Samuel called one Eagles cornerback “trash” – and then entered the Linc as a rare betting favorite against
a 10-win team at home.

The Eagles (10-2) suffered a scare when Hurts jogged off the field in the fourth and went straight to the locker room. He was cleared to play and returned to
action with about 10 minutes left. It was not immediately clear when Hurts was injured.

By the time last season’s NFL MVP runner-up returned, the Eagles trailed 35-13 and were on their way toward their first home loss of the season. Hurts did hit
DeVonta Smith for a two-yard TD on his return drive. He finished 26-of-45 passing for 298 yards. Purdy put the finishing touches on the win with a 46-yard TD to
Samuel.

Perhaps all that braggadocio paid off for the 49ers.
</item 38>

<item 39>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
For Detroit, it's all about courage. For a team that was once down and out, much like their city, they are now once again on the rise. The Lions are supposed to
be the kings of the jungle and hope to become the NFL's version in 2023. In terms of the food chain, a basic concept taught in science classes across the
country, lions are the hunter rather than the hunted, but it's been the other way around for far too long, as 31 other teams have feasted on a once proud
franchise.

Their motors have been restored and the team has been rebuilt with an attitude and moxie that would make even the most "football guy" of "football guys" blush.
It's a combination of quiet confidence and a downright arrogant confidence that draws people in to root for -- or against -- an underdog.

While the Packers have packed up the Lions for years, that all began to change in 2022.
</item 39>

<item 40>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
The Cowboys' offense looked very much like it did in last season's playoff exit -- uncomfortable and out of sorts. Dallas was sluggish through the first half of
the game, and the night unraveled quickly when Dak Prescott threw three interceptions in the second half. It got out of hand so quickly that Sam Darnold entered
the game for the 49ers with more than 12 minutes remaining.

For Prescott and the Cowboys, questions about whether they can get over the hump against the class of the NFC will persist. While their wins have been dominant,
Dallas already sits two games behind the Eagles in the NFC East.

The 49ers are rolling right along, with all of their major offensive weapons now healthy and the defense living up to its billing as well. San Francisco, now
5-0, doesn't face a team with a winning record until Week 10, though the Browns, Vikings and Bengals are far from automatic wins for Kyle Shanahan's team.
</item 40>

<item 41>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Good drive by Green Bay, which finally shows signs of life after a putrid first half.

Lions 27, Packers 11

10:05 p.m.: Great throw by Love, who threads the needle to find Romeo Doubs to keep the chains moving on fourth down. Packers on the brink of a much-needed
touchdown.

9:58 p.m.: This Green Bay offense has to come to life sooner rather than later. If not, this scoreboard could get even more crooked.

Good start to the third frame by the Pack, who find themselves marauding down the field with relative ease in the quarter's opening possession. Every journey
began with a single step...

9:20 p.m.: INTERCEPTION! Love's errant pass gets snagged by Jerry Jacobs, who plods 20 yards back to put Detroit on the brink of the promised land. It's been a
putrid start for Love and the Green Bay offense so far.

9:15 p.m.: FIELD GOAL!
</item 41>

<item 42>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
11:00 p.m. INTERCEPTION — So much for that. Oren Burks gets the 49ers' third interception of the night as Prescott continues to struggle against this San
Francisco defense.

10:59 p.m. — The Cowboys are moving the ball now, when it's a bit too late. Tony Pollard rips off a 26-yard run to finally break through.

49ers 42, Cowboys 10

10:53 p.m. TOUCHDOWN — And just like that, Jordan Mason runs in for another 49ers touchdown! This is ugly on all levels for the Cowboys, and Leighton Vander
Esch is now down as well. 42-10, 49ers.

10:52 p.m. INTERCEPTION — Fred Warner now has a sack, forced fumble and an interception tonight. A pass from Prescott is deflected into his hands, and the 49ers
have another instant scoring opportunity.
</item 42>

<item 43>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
But the reeling defense and pass protection continue to be un-winnable assets.

24. Atlanta Falcons 4-5 (20)

The Falcons were forced into another QB change because of injury. They found a few more answers offensively but that position still made it an overall mess. The
defense also doesn't seem to be well prepared to different styles of offenses.

25. Green Bay Packers 3-6 (25)

The Packers saw some improved work from Jordan Love and his young weapons as they spread the ball well in the passing game, but the run defense woes and key
holes in coverage have all but doomed their playoff hopes.

26. Los Angeles Rams 3-6 (26)

The Rams are hoping to find a few more answers during a bye, as Sean McVay has seen his good coaching job earlier in the season go by the wayside with key
offensive injuries.
</item 43>

<item 44>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
But Green Bay is firmly back in it.

Lions 27, Packers 17

End of third quarter: Lions 27, Packers 11

10:31 p.m.: Impressive pitch-and-catch by Love and Jayden Reed. Packers hurler drops it right into Reed's bread basket for the 44 yard gain as the quarter comes
to an end.

10:24 p.m.: Things slow down for Green Bay on Drive No. 2 of the second half. Punt incoming. Packers need less of that as they attempt their second-straight
rally in as many weeks.

10:15 p.m.: Green Bay defense firms up, forcing yet another punt. Can Love and the Packers bring things back?

10:10 p.m.: TOUCHDOWN! Love finds Watson for the TD, giving Packers fans a much-needed boost of adrenaline. Good drive by Green Bay, which finally shows signs
of life after a putrid first half.
</item 44>

<item 45>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Thursday's game appeared to be destined to be a showcase of quarterbacking excellence; Jared Goff and Jordan Love have proved themselves to be formidable
options under center since taking the reins of their respective offenses.

But it was the supporting cast that proved so pivotal to Thursday's result for Dan Campbell's side. David Montgomery was a nuisance, slipping through crevices
to puncture the heart of the Packers' defense. When it wasn't him, it was that receiving corp, made up of unheralded youngsters who have all come up trumps in
recent weeks.

As for Love and Co., Thursday's affair was another indication of just how much work there is to be done before they compete for the divisional crown. There were
certainly some flits of delight, throws that highlighted just how much talent Love possesses in his golden arm.

But there were low points, too, misshapen efforts that fell harmlessly to the turf, or harmfully into the opposition's hands.
</item 45>

<item 46>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
1 seed is undoubtedly in the back of their minds.

6:45 p.m. — The run of great prime-time weather continues, as it's set to be in the high 70s and sunny in Santa Clara tonight. No excuses for these powerful
offenses.

What channel is Cowboys vs. 49ers on today?

Game: Cowboys vs. 49ers

Cowboys vs. 49ers Date: Sunday, Oct. 8

Sunday, Oct. 8 TV channel: NBC

NBC Live stream: Peacock | NBCSports.com | Fubo (U.S.) | DAZN (Canada)

Cowboys vs, 49ers can be watched on NBC, which has long held the broadcast rights to "Sunday Night Football."

Mike Tirico (play-by-play) and Cris Collinsworth (color analyst) will be on the call from Levi's Stadium while Melissa Stark will serve as the sideline
reporter.
</item 46>

<item 47>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Sideline reporter Kaylee Hartung will offer updates throughout the affair.

Fans in the Detroit and Green Bay markets can watch the Amazon broadcast over the air on local channels.

Those in Canada can stream it on DAZN, which carries every NFL game all season.
</item 47>

<item 48>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
Seahawks vs. Giants final score

1 2 3 4 Final Seahawks 7 7 7 3 24 Giants 0 3 0 0 3

Seahawks vs. Giants live updates, highlights from 'Monday Night Football'

(All times Eastern)

Team stats

Seahawks Giants First downs 13 17 Third downs 3-12 6-16 Fourth downs 0-2 2-4 Total yards 281 248 Passing 15-26, 160 yards 27-34, 136 yards Rushing att-yards
23-121 29-112 Penalties-yards 8-74 6-45 Turnovers 0 3 Time of possession 24:00 36:00

Final: Seahawks 24, Giants 3

11:18 p.m.: It's a wrap at MetLife Stadium. The Seahawks advance to 3-1 on the year while the Giants fall to 1-3.
</item 48>

<item 49>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
: The Giants just pulled off one of their biggest plays of the night.

On fourth-and-1 from Seattle's 16-yard line, Brian Daboll elects to keep the offense on the field instead of electing for three points. Out of play action,
Daniel Jones rolls out and gets 11 yards to bring New York within five yards of its first touchdown of the night.

10:22 p.m.: A 53-yard field goal attempt from Jason Myers is no good!

Seattle fails to add to its lead and New York gets another chance to cut into its deficit, this time, with field position that's slightly more favorable. The
Giants will take over at their on 34-yard line.

10:16 p.m.: New York's drive ends about as quickly as it began as it is three-and-out for the Giants.

Daniel Jones is sacked for a sixth time this evening. This time it's eight-time Pro Bowler Bobby Wagner.
</item 49>

<item 50>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
MORE: Why Brock Purdy remains one of the NFL's cheapest starting quarterbacks

The Sporting News tracked live scoring updates and highlights from "Sunday Night Football" between the Cowboys and 49ers. Check out all of the key moments you
might have missed.

Cowboys vs. 49ers final score

1 2 3 4 F Cowboys 0 7 3 0 10 49ers 7 14 7 14 42

Cowboys vs. 49ers results, highlights from 'Sunday Night Football'

(All times Eastern)

Final: 49ers 42, Cowboys 10

11:26 p.m. — That'll do it! The 49ers trounce the Cowboys and reassert themselves as one of the NFL's most dominant teams. Through five games, San Francisco is
unbeaten.

11:24 p.m. — These teams are trading possessions as the clock winds down.
</item 50>
=== END ITEM id=85448776fddfbe41 ===
