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

=== ITEM id=1dbeeb5f62124174 ===
Gold sentences and facts:
Sentence 0: Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
  Fact 0.0: Google unveiled new hardware.
  Fact 0.1: Google held its annual hardware event this week.
  Fact 0.2: Google is a search and consumer tech giant.
  Fact 0.3: At its annual hardware event, Google showed off what it has been working on.
Sentence 1: Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part.
  Fact 1.0: Wired removed the article for not meeting its standards.
  Fact 1.1: The article was shared widely on social media.
  Fact 1.2: The article was boosted in write-ups at other outlets.
  Fact 1.3: The extent to which the article was shared and boosted shows how willing people are to accept foul play on Google's part.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content.
  Fact 2.2: The case argues that Google "siphons off" news publishers' readers.
  Fact 2.3: The case argues that Google "siphons off" news publishers' ad revenue.
  Fact 2.4: The case argues that Google obtains these through anticompetitive means.

Retrieved items:
<item 1>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 7>

<item 8>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
</item 8>

<item 9>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Google’s software is constantly crawling the web and updating its central index, which itself is more than 100 million gigabytes in size. When you enter a
query, Search uses a few quick processes to decide the meaning of your question, including what kind of information you might be after. Then, beyond just
looking for sites with the words you typed in, the system uses machine learning to rank results according to what it thinks you’re looking for, prioritising
high quality and easy to use sites. If you allow it, Google does bring in some information it knows about you to further refine results, including your location
and web history. If you’re looking to turn this off, it’s in your Google account settings called “Web & App Activity”. Google says it does not infer personal
information like race or religion to shape results. Loading Google works with independent “search quality raters” to gauge the effectiveness of its processes,
and says it ran almost 900,000 quality tests in 2022.
</item 9>

<item 10>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 10>

<item 11>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 11>

<item 12>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 12>

<item 13>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Looking back on all that had changed around what first attracted people to Google, from the introduction of ads to the efforts to keep users within the universe
of Google products, this seemed to be the last straw.

The second major reason why Google results feel different lately was, of course, SEO — specifically, the obnoxious-kid-refusing-to-behave-in-class kind of SEO.

“SEO that goes against Google’s guidelines, it’s not new,” Ray explained. A decade ago, it used to be called “black hat” SEO, in comparison to the search
engine-approved “white hat” tactics. And Google has, as Sullivan and many SEOs told me, gotten better over time at catching SEOs playing tricks on the
algorithm.
</item 13>

<item 14>
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
</item 14>

<item 15>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 15>

<item 16>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 16>

<item 17>
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
</item 17>

<item 18>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 18>

<item 19>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 19>

<item 20>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 20>

<item 21>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 21>

<item 22>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 22>

<item 23>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 23>

<item 24>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 24>

<item 25>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 25>

<item 26>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 26>

<item 27>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 27>

<item 28>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 28>

<item 29>
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
</item 29>

<item 30>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.
</item 30>
=== END ITEM id=1dbeeb5f62124174 ===

=== ITEM id=77601b1fdc5060dc ===
Gold sentences and facts:
Sentence 0: In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Sorcerer builds for season 2.
  Fact 0.0: The guide is a Diablo 4 guide.
  Fact 0.1: The guide gathers the best Diablo 4 Sorcerer builds for season 2.
  Fact 0.2: The guide simplifies the best Diablo 4 Sorcerer builds for season 2.
Sentence 1: In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Barbarian builds for season 2.
  Fact 1.0: The guide is a Diablo 4 guide.
  Fact 1.1: The guide gathers the best Diablo 4 Barbarian builds for season 2.
  Fact 1.2: The guide simplifies the best Diablo 4 Barbarian builds for season 2.

Retrieved items:
<item 1>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
this build, you’ll want to take One-Handed Mace in your Technique.

Vampiric Powers for Double Swing Barbarian build

Season 2 adds the new Vampiric Powers, which allow you to slot five additional abilities into your character to give you new effects. There are 22 total
Vampiric Powers, and each are class agnostic, meaning they’re not unique to any one class.

AaronActionRPG recommends these five Vampiric Powers:

Anticipation

Prey on the Weak

Ravenous

Domination

Metamorphosis

To insert those powers you’ll need to have these Pacts available on your armor:

8 Ferocity

3 Divinity

2 Eternity

Legendary Powers and Uniques for Double Swing Barbarian build

To make this build work, you’re going to need powerful Legendary Powers and Unique items to augment your build.

AaronActionRPG recommends these items:

Helm : Weapon Master’s Aspect

: Weapon Master’s Aspect Chest : Aspect of Numbing Wrath (dungeon drop)

: Aspect of
</item 1>

<item 2>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Best Sorcerer leveling build

This build is based off Lexyu’s Sorcerer leveling guide at IcyVeins, and is purely for leveling: i.e., allocating your 58 skill points between levels 1 and 50.
If you want more detailed information about the item affixes you want to look for on your gear, or which other Legendary powers are great for your build, check
out the original IcyVeins guide.

Best skills for leveling

This Sorcerer build guide is going to see you harness the awesome power of Lightning to speed through the leveling process. There are a few options for Core
skills here, but we’re specifically going to be showing you Lexyu’s Chain Lightning variation. In this build, you’ll also use Frost Bolt, Teleport, Frost Nova,
Lightning Spear, and Ice Armor.

Here are the best skills for a Sorcerer leveling build, organized by the exact order you should purchase them in — bearing in mind that some skills require you
to allocate a number of skill points in the tree in order to unlock them. As such,
</item 2>

<item 3>
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
The Echo of Duriel (who is level 100) will be the most difficult of the new bosses, but the reward will obviously be worth the struggle, considering he’s the
only way for players to farm Uber Uniques in Diablo 4. For players who have more in-game gold than time, Blizzard also confirmed in our interview that players
will be able to trade with one another to get the items needed to summon Duriel and the other bosses.

While these bosses are a form of milestone activity for players, Nightmare Dungeons will still exist as places to both farm for better items and push builds to
their limit. However, the dungeons are becoming much simpler and less grindy. The design team has reworked several Nightmare Dungeon affixes to make them less
obnoxious, and Nightmare Sigils now teleport players directly into the dungeon instead of to the door.
</item 3>

<item 4>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Best Barbarian leveling build

This build is based off Lexyu’s Barbarian leveling guide at IcyVeins, and is purely for leveling: i.e., allocating your 58 skill points between levels 1 and 50.
If you want more detailed information about the item affixes you want to look for on your gear, or which other Legendary powers are great for your build, check
out the original IcyVeins guide.

Best skills for leveling

Most Barbarian leveling guides you’ll find make use of Hammer of the Ancients as your primary skill, as it deals massive damage while remaining very cheap to
cast. In this build, you’ll also use Frenzy, Death Blow, Rallying Cry, Leap, and Wrath of the Berserker.

Here are the best skills for a Barbarian leveling build, organized by the exact order you should purchase them in — bearing in mind that some skills require you
to allocate a number of skill points in the tree in order to unlock them. As such, you’ll need to unlock new skills before enhancing existing
</item 4>

<item 5>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Weapon Master’s Aspect Chest : Aspect of Numbing Wrath (dungeon drop)

: Aspect of Numbing Wrath (dungeon drop) Gloves : Aspect of Anemia (dungeon drop)

: Aspect of Anemia (dungeon drop) Pants : Aspect of Might (dungeon drop)

: Aspect of Might (dungeon drop) Boots : Exploiter’s Aspect

: Exploiter’s Aspect Two-handed Bludgeoning : Aspect of Retribution (dungeon drop)

: Aspect of Retribution (dungeon drop) Two-handed Slashing : Accelerating Aspect

: Accelerating Aspect Dual Wield 1 : Edgemaster’s (dungeon drop)

: Edgemaster’s (dungeon drop) Dual Wield 2 : Aspect of Berserk Ripping (dungeon drop)

: Aspect of Berserk Ripping (dungeon drop) Amulet : Earthquake Aspect

: Earthquake Aspect Ring 1 : Aspect of Berserk Fury

: Aspect of Berserk Fury Ring 2: Aspect of Limitless Rage

If you want to get into the nitty gritty
</item 5>

<item 6>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Ring 2: Aspect of Limitless Rage

If you want to get into the nitty gritty of stat priorities, refer to AaronActionRPG’s original guide.

Paragon Boards for Double Swing Barbarian build

The Paragon Board system is very complex, and you’ll want to carefully pair your Glyphs with specific boards for your class:

Starter Board / Bloodfeeder Glyph Decimator / Ambidextrous Glyph Carnage / Might Glyph Flawless Technique / Ire Glyph Weapon Master / Territorial Glyph Bone
Breaker / Exploit Glyph Warbringer / Wrath Glyph

You should level your Glyphs to 15 in this specific order:

Exploit Territorial Ambidextrous Might Wrath Bloodfeeder Ire

You can path your own way through the Paragon Boards following these basic guidelines. However, for the best results, follow AaronActionRPG’s path exactly.

How to make a great Barbarian build

Barbarians are Diablo 4’s only pure melee class, with the other classes having hybrid builds for multiple ranges.
</item 6>

<item 7>
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
Shely told the panel that last season, players set a goal for themselves: Get to level 100. However, many players found that they were never able to hit that
milestone because XP gains were so slow. This change should allow more players to reach the level cap each season, allowing them to play with a truly maxed-out
build.

As for how players will spend their time at max level and where they’ll be able to earn some of the game’s most powerful items, enter the five new endgame
bosses: Grigoire The Galvanic Saint, the Echo of Varshan, The Beast in the Ice, the Dark Master, and the Echo of Duriel. Each boss has its own laundry list of
busy work for players to complete in order to unlock their encounter — such as completing Nightmare Dungeons above level 30 or completing Grim Favors — but they
also come with Unique rewards.
</item 7>

<item 8>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
refer to Northwar’s original guide.

Paragon Boards for Sorcerer Firewall build

The Paragon Board system is very complex, and you’ll want to carefully pair your Glyphs with specific boards for your class:

Starter Board / Reinforced Glyph Burning Instinct / Destruction Glyph Searing Heat / Adept Glyph Static Surge / Control Glyph Enchantment Master / Exploit

You should level your Glyphs to 15 in this specific order:

Destruction Control Adept Exploit Reinforced

You can path your own way through the Paragon Boards following these basic guidelines. However, for the best results, follow Northwar’s path exactly.

How to make a great Sorcerer build

Sorcerers are Diablo 4’s pure caster class, and use elemental magic to differentiate from the spells of the Druid and Necromancer. The Sorcerer has six Key
Passives, but only three elements, which is what you’ll base your build around:

Lightning

Fire

Ice

Lightning builds utilize Lightning skills to deal with large groups of enemies at once.
</item 8>

<item 9>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Northwar recommends these items:

Helm : Shared Misery (dungeon drop) or Godslayer Crown (Unique)

: Shared Misery (dungeon drop) or Godslayer Crown (Unique) Chest : Snowveiled (dungeon drop) or Raiment of the Infinite (Unique)

: Snowveiled (dungeon drop) or Raiment of the Infinite (Unique) Gloves : Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop)

: Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop) Pants : Disobedience (dungeon drop)

: Disobedience (dungeon drop) Boots : Ghostwalker (dungeon drop), Bounding Conduit (dungeon drop), Flickerstep (Unique)

: Ghostwalker (dungeon drop), Bounding Conduit (dungeon drop), Flickerstep (Unique) Wand : Edgemaster’s (dungeon drop), Conceited,
</item 9>

<item 10>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
When making a Lightning build, you’ll end up using the Overflowing Energy Key Passive or the Vyr’s Master Key Passive, depending on if you’re in melee range or
not.

Ice builds specialize in slowing enemies down, making it easier to survive their attacks. Ice builds will take advantage of either the Avalanche Key Passive or
the Shatter Key Passive, depending on how much your build relies on crowd control

Fire builds are all about big explosions and burning your enemies. For these builds, you’ll typically use either the Combustion Key Passive for burning enemies
or the Esu’s Ferocity Key Passive for chunky critical strikes.

When making your own build for a Sorcerer, you’ll likely end up sticking to one element for your main damage sources. But that’s not the most important thing
you need to remember. Sorcerers die very, very easily in Diablo 4, and they have the weakest defensive stats by far — however, they can deal a massive amount of
damage when built correctly.
</item 10>

<item 11>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Recharging Aspect (Zenith in Fractured Peaks) — Each time Chain Lightning bounces off you, gain X Mana

(Zenith in Fractured Peaks) — Each time Chain Lightning bounces off you, gain X Mana Aspect of Control (Sunken Library in Kehjistan) — You deal X% more damage
to Immobilized, Stunned, or Frozen enemies

(Sunken Library in Kehjistan) — You deal X% more damage to Immobilized, Stunned, or Frozen enemies Edgemaster’s Aspect (Oldstones dungeon in Scosglen) — Skills
deal X% increased damage based on Primary Resource when cast, receiving the maximum benefit while you have full Primary Resource

Sorcerer Season 2 endgame build — Firewall

This build is based off of Northwar’s Sorcerer endgame guide at MaxRoll.gg, and is a general endgame guide capable of finding success in all activities.
</item 11>

<item 12>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
This guide assumes that you’re over level 50, have access to all 58 skill points, and are making progress on your Paragon Boards. If you want more detailed
information about the item affixes you want to look for on your gear, or why these Legendary powers are great for your build, check out the original MaxRoll
guide.

Best skill points for Sorcerer Firewall endgame build

This build utilizes Firewall as its core skill and main means of dealing damage. It also uses Fire Bolt, Inferno, Teleport, Ice Armor, and Flame Shield.

Since you need all 58 points for this build, the order you select these skills in doesn’t matter as long as you have the pre-requisite points in the earlier
Skill Tree tiers.
</item 12>

<item 13>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
As such, you’ll need to unlock new skills before enhancing existing ones:

Frenzy / Enhanced Frenzy Hammer of the Ancients / Enhanced Hammer of the Ancients / Furious Hammer of the Ancients Combat Frenzy Rallying Cry / Enhanced
Rallying Cry / Tactical Rallying Cry Hammer of the Ancients (ranks 2-3) Leap / Enhanced Leap / Power Leap Hammer of the Ancients (ranks 4-5) Death Blow /
Enhanced Death Blow / Warrior’s Death Blow / Death Blow (rank 2-5) Wrath of the Berserker / Prime Wrath of the Berserker / Supreme Wrath of the Berserker Pit
Fighter (ranks 1-3) Tempered Fury Furious Impulse (ranks 1-3) Walking Arsenal Hamstring No Mercy Expose Vulnerability (ranks 1-3) No Mercy (ranks 2-3) Swiftness
(ranks 1-3) Thick Skin (ranks 1-3) Counteroffensive (ranks 1-3) Defensive
</item 13>

<item 14>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
The idea here is that you’re constantly swapping between your skills and weapons to give yourself massive damage buffs.

When making a build, you want to have one of these Key Passives in mind and choose your skills accordingly. Thankfully skills that share certain aspects (like
Bleed) are color-coded on the skill tree, and you can even use the in-game search function to find them all. Other Barbarian builds let you take advantage of
mechanics like Thorns or Shouts. But they still typically end with one of these Key Passives.

Looking for other Diablo 4 builds? We have explainers on the Sorcerer, Rogue, Druid, and Necromancer classes.
</item 14>

<item 15>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
and is a general endgame guide capable of finding success in all activities. This guide assumes that you’re over level 50, have access to all 58 skill points,
and are making progress on your Paragon Boards. If you want more detailed information about the item affixes you want to look for on your gear, or why these
Legendary powers are great for your build, check out the original IcyVeins guide.

Best skill points for Double Swing Barbarian endgame build

This build utilizes Double Swing as its core skill. It also uses Steel Grasp, Ground Stomp, Iron Maelstrom, Charge, and Lunging Strike.

Since you need all 58 points for this build, the order you select these skills in doesn’t matter as long as you have the pre-requisite points in the earlier
Skill Tree tiers. Here is how you should spend your Skill Points for this build:

Lunging Strike / Enhanced Lunging Strike / Combat Lunging Strike

Double Swing (rank 1-5) / Enhanced Double Swing / Furious Double Swing

Endless
</item 15>

<item 16>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Swing (rank 1-5) / Enhanced Double Swing / Furious Double Swing

Endless Fury

Ground Stomp (rank 1-5) / Enhanced Ground Stomp / Strategic Ground Stomp

Charge (rank 1-5) / Enhanced Charge / Power Charge

Aggressive Resistance (rank 1-3)

Prolific Fury (rank 1-3)

Steel Grasp / Enhanced Steel Grasp / Fighter’s Steel Grasp

Thick Skin

Defensive Stance (rank 1-3)

Counteroffensive (rank 1-3)

Pit Fighter (rank 1-3)

No Mercy (rank 1-3)

Iron Maelstrom / Prime Iron Maelstrom / Supreme Iron Maelstrom

Duelist (rank 1-3)

Wallop

Concussion (rank 1-3)

Walking Arsenal

Arsenal System for the Double Swing Barbarian build

For this build, you’ll want to take One-Handed Mace in your Technique.
</item 16>

<item 17>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Barbarians are a relatively simple class in Diablo 4: They smash stuff. However, Diablo 4 is underscored by math that’s so complex that even the simple act of
smashing stuff becomes a dizzying equation. That’s where we come in, with our suggestions of the best Barbarian builds for Diablo 4 season 2.

Barbarians are big, muscular, and powerful. They excel at using Bleed to kill their enemies over time. And their unique weapon-swapping system means they get to
have more Legendary powers at their disposal than any other class.

In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Barbarian builds for season 2. If you need even more information on item affixes or
other more complex ideas, we’ve been sure to link out to the more detailed versions of these builds, as well as where to find their original creators.
</item 17>

<item 18>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
There are four main ideas that all Barbarian builds stem from:

Fury dumping (most popular in Diablo 4’s pre-season)

Weapon swapping

Bleed

Berserk

Fury dump builds make use of the Unbridled Rage Key Passive, which causes your Fury spenders to cost double the Fury but deal more than double the damage. These
kinds of builds obviously require very steady Fury generation, as it’s easy to starve yourself. Berserk builds that use the Unconstrained Key Passive are all
about keeping your Barbarian in the Berserk stage at all times, which nets you a hefty bonus. All Barbarian builds will likely use some skills that make them go
Berserk, the Unconstrained Key Passive just enhances the effect.

Bleed builds use the Gushing Wounds Key Passive to create a critical strike heavy build that’s great for whittling down groups of enemies with damage over time
and explosions. Weapon swapping builds offer what is almost certainly the most unique playstyle for Barbarians and takes advantage of the Walking Arsenal Key
Passive.
</item 18>

<item 19>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
If you don’t want to end up dead more often than not, make sure to invest heavily in defensive skills while making your way toward one of the six Key Passives
we listed above.

Looking for other Diablo 4 builds? We have explainers on the Barbarian, Rogue, Druid, and Necromancer classes.
</item 19>

<item 20>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
dealing X% of its damage to enemies

(Sunken Ruins dungeon in Scosglen) — Hammer of the Ancients quakes outwards, dealing X% of its damage to enemies Aspect of the Expectant (Underroot dungeon in
Scosglen) — Attacking enemies with a Basic Skill increases the damage of your next Core Skill cast by X% up to X%

(Underroot dungeon in Scosglen) — Attacking enemies with a Basic Skill increases the damage of your next Core Skill cast by X% up to X% Edgemaster’s Aspect
(Oldstones dungeon in Scosglen) — Skills deal X% increased damage based on Primary Resource when cast, receiving the maximum benefit while you have full Primary
Resource

Barbarian Season 2 endgame build — Double Swing

This build is based off of AaronActionRPG’s Double Swing Barbarian endgame guide at IcyVeins, and is a general endgame guide capable of finding success in all
activities.
</item 20>

<item 21>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
(ranks 1-3) Counteroffensive (ranks 1-3) Defensive Stance (ranks 1-3) Slaying Strike (ranks 1-3) Aggressive Resistance Prolific Fury

Best Weapon Expertise setup for leveling

Once you unlock the Weapon Expertise system for your Barbarian, you’ll want to place the two-handed sword in your Technique slot. This will cause all of your
Hammer of the Ancients damage to make enemies Bleed.

Best Legendary Aspects for Barbarian leveling build

When you first start your new character, complete these three dungeons first in order to collect their Aspects. Then place them on a piece of gear (ideally
jewelry, when possible). Just keep in mind that you should replace these Legendaries with new versions using the imprint system every few levels.

Aspect of Ancestral Force (Sunken Ruins dungeon in Scosglen) — Hammer of the Ancients quakes outwards,
</item 21>

<item 22>
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
All of these changes, as well as the content for Season of Blood, will come to the game on Oct. 17, just a few weeks before fans are expecting Blizzard to
announce Diablo 4’s first expansion at BlizzCon 2023.
</item 22>

<item 23>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
As such, you’ll need to unlock new skills before enhancing existing ones:

Frost Bolt / Enhanced Frost Bolt Chain Lightning Glinting Frost Bolt Enhanced Chain Lightning / Greater Chain Lightning Teleport Frost Nova / Enhanced Frost
Nova / Mystical Frost Nova Chain Lightning (rank 2-5) Fireball (rank 1 only for the Enchantment slot) Lightning Spear Glass Cannon (rank 1-3) Devastation
Elemental Dominance (rank 1-3) Coursing Currents (rank 1-3) Elemental Attunement Ice Armor / Enhanced Ice Armor / Shimmering Ice Armor Enhanced Teleport /
Shimmering Teleport Enhanced Lightning Spear Vyr’s Mastery Invoked Lightning Spear / Lightning Spear (rank 2-5) Frost Bolt (rank 2-5) Conduction (rank 1-3)
Electrocution (rank 1-3) Align the Elements Protection (rank 1-3) Icy Veil (rank 1-3) Frost Nova (rank 2-3)

Best Enchantment setup for leveling

The Enchantment
</item 23>

<item 24>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Sorcerers are the magic-slinging wizards of Sanctuary in Diablo 4. But with each element they can harness bringing its own unique effect, it can be easy to get
overwhelmed when trying to design a Sorcerer build. That’s where we come in, with our suggestions of the best Sorcerer builds for Diablo 4 season 2.

Sorcerers are capable of harnessing lightning to deal with crowds, ice to keep the hordes at bay, and fire to burn the demons of hell. But they’re also able to
enchant their own spells with unique effects, intricately combining abilities for devastating results.

In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Sorcerer builds for season 2. If you need even more information on item affixes or other
more complex ideas, we’ve been sure to link out to the more detailed versions of these builds, as well as where to find their original creators.
</item 24>

<item 25>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
you’ll want to use these spells for your two Enchantment Slots:

Firewall Enchantment

Fireball

Vampiric Powers for Sorcerer Firewall build

Season 2 adds the new Vampiric Powers, which allow you to slot five additional abilities into your character to give you new effects. There are 22 total
Vampiric Powers, and each are class agnostic, meaning they’re not unique to any one class.

Northwar recommends these five Vampiric Powers:

Anticipation

Prey on the Weak

Ravenous

Flowing Veins

Metamorphosis

To slot those powers you’ll need to have these Pacts available on your armor:

9 Ferocity

5 Divinity

4 Eternity

Legendary Powers and Uniques for Sorcerer Firewall build

To make this build work, you’re going to need powerful Legendary Powers and Unique items to augment your build.
</item 25>

<item 26>
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
As for gems, these items will no longer take up bag space, as they’re now considered crafting materials. This change comes alongside a massive upgrade to the
Stash itself. Stashes now appear in more locations — including all minor towns with a Waypoint in them — and players can search their Stashes for specific items
and affixes.

These are just a handful of the major changes coming to Diablo 4 with Season of Blood. Shely and Piepiora told us that the focus on quality-of-life upgrades was
largely due to season 2 being the first season where they and the rest of the Diablo 4 team could really listen to feedback. Because of how close Season of the
Malignant launched after the release of Diablo 4 (just over a month), that season’s content was largely finished before players had really gotten to dive into
the action-RPG. However, with season 2, Blizzard has been able to react more to player feedback and finally create some sweeping, specific changes that players
have been requesting.
</item 26>

<item 27>
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
Diablo 4’s Season of Blood brings some cool new vampire powers into the fold, but the real star of the action game’s second season is its sweeping
quality-of-life changes. Players will now have an easier time leveling up, gaining quality loot, and just navigating the open world of Sanctuary.

Game director Joe Shely and associate game director Joe Piepiora sat down to do a roundtable interview with Polygon and other outlets ahead of Blizzard’s Oct.
4th livestream, and gave additional context to Nightmare Dungeon changes, Renown, and more.

One of the most notable changes for players in Diablo 4’s second season update will be an increase in experience gains. Blizzard claims players will be able to
reach the level cap of 100 about 40% faster than in Season of the Malignant — which already increased experience gains compared to Diablo 4’s launch. This
process should be much smoother for players as well, as monsters will now match their levels between 55 and 75.
</item 27>

<item 28>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Conceited, or Storm Swell (dungeon drop)

: Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop) Focus : Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop)

: Edgemaster’s (dungeon drop), Conceited, or Storm Swell (dungeon drop) Amulet : Control (dungeon drop)

: Control (dungeon drop) Ring 1 : Prodigy’s (dungeon drop) or Retribution (dungeon drop)

: Prodigy’s (dungeon drop) or Retribution (dungeon drop) Ring 2: Engulfing Flames or X’Fal’s Corroded Signet (Unique)

If you want to get into the nitty gritty of stat priorities, refer to Northwar’s original guide.
</item 28>

<item 29>
Diablo 4 season 2’s biggest draw is its quality-of-life updates (Polygon, 2023-10-04T20:01:00+00:00)
Previously, players would have to max out their Renown — which all but required doing nearly every dungeon and most side quests in the game — every season in
order to get all available Skill and Paragon Points. Not only is that not the case anymore, but (unlike with the Altars of Lilith carryover) you won’t need to
first log into a character that’s completed the Renown track in order to propagate that data to the rest of your account. Piepiora told us that all of that data
migration should happen without that extra step in season 2.

This is yet another massive 180 on the Renown system from Blizzard in barely five months’ time. When we asked about the massive shift, Piepiora told us that
players simply weren’t having fun redoing the grind every season, hence the move here (as with the Altars last season) to alleviate the pressure on players to
only play “efficiently.”
</item 29>

<item 30>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Here is how you should spend your Skill Points for this build:

Fire Bolt / Enhanced Fire Bolt / Glinting Fire Bolt

Fireball / Enhanced Fireball / Greater Fireball

Potent Warding (rank 1-3)

Devastation

Elemental Dominance (rank 1-3)

Flame Shield / Enhanced Flame Shield / Shimmering Flame Shield

Teleport (rank 1-4) / Enhanced Teleport / Shimmering Teleport

Ice Armor/Enhanced Ice Armor

Glass Cannon (rank 1-3)

Precision Magic (rank 1-3)

Align the Elements

Mana Shield (rank 1-3)

Protection (rank 1-3)

Firewall (rank 1-5) / Enhanced Firewall / Wizard’s Firewall

Inner Flames (rank 1-3)

Crippling Flames (rank 1-3)

Inferno/Prime Inferno/Supreme Inferno

Fiery Surge

Endless Pyre

Combustion

Enhancement slots for Sorcerer Firewall build

For this build,
</item 30>
=== END ITEM id=77601b1fdc5060dc ===

=== ITEM id=59881071d587554e ===
Gold sentences and facts:
Sentence 0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 0.0: Huang said this during cross-examination.
  Fact 0.1: Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 0.2: Huang said Paradigm was told Bankman-Fried did not want investors as directors.
  Fact 0.3: Huang said Bankman-Fried did plan on having a board with experts.
Sentence 1: The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the instruction of SBF.
  Fact 1.0: The second week of the trial included standout testimony.
  Fact 1.1: The standout testimony came from Caroline Ellison.
  Fact 1.2: Caroline Ellison is Alameda Research's former CEO.
  Fact 1.3: Caroline Ellison claimed she took $14 billion from customers.
  Fact 1.4: Caroline Ellison claimed she took the $14 billion to repay debts to lenders.
  Fact 1.5: Caroline Ellison claimed she used the money as a line of credit.
  Fact 1.6: Caroline Ellison claimed she did this under the instruction of SBF.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense argued that the FTX founder acted in good faith.
  Fact 2.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“You will see the full picture.”

…or a well-meaning founder?

But Bankman-Fried, whose cheekbones were more prominent after spending about seven weeks in a Brooklyn prison, was no liar, according to Mark Cohen, one of his
lawyers. “Sam didn’t defraud anyone,” he said early on in his opening statement.

What the jury will see is a nerdy startup founder who acted in “good faith,” not the prosecution’s “cartoon of a villain.” (Cohen repeatedly harped on
Bankman-Fried’s allegedly good-faith actions throughout his address to the jury.)

Alameda was not subterranean or shady. It was a successful hedge fund, he said. FTX was no Ponzi scheme. It was a “very innovative, successful company.”
</item 1>

<item 2>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 2>

<item 3>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
Meanwhile, Bankman-Fried tweeted. “FTX is fine. Assets are fine” and “We don’t invest customer assets even in treasuries.”

Pointing at Bankman-Fried, Rehn said, “This man stole billions of dollars from thousands of people.”

So how was the defense going to follow it up? I was very curious, having learned yesterday that Bankman-Fried had never been offered a plea deal since he and
his attorneys had told the government they wouldn’t negotiate. Surely there would be some manner of evidence, some something, that would have made him so
confident.

There was, instead, a metaphor.

Defense attorney Mark Cohen, with the energy of a patient father telling his obnoxious children a bedtime story, assured us that working at a startup was like
building a plane while flying it, and that FTX the plane had flown right into the perfect storm: the crypto crash.
</item 3>

<item 4>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
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
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.
</item 6>

<item 7>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 7>

<item 8>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
</item 8>

<item 9>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 9>

<item 10>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 10>

<item 11>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Rather, he was willing to give up his personal wealth to make customers whole, Cohen argued.

“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”

A Frenchman who lives in London testifies in New York

After lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.

The first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong
French accent, decided to invest in crypto and landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.

On Nov. 8, in the crypto exchange’s final few days, he tried to pull out his cash and crypto.
</item 11>

<item 12>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 12>

<item 13>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“Sam acted in good faith and took reasonable business measures” is a pretty hard pill to swallow with that in mind.

Man, it’s no good when your defense lawyer has just made you sound worse than the prosecution already did. And while Cohen tried to make the common white-collar
defense argument that Bankman-Fried, as CEO, was simply too busy to oversee what everyone did every day, he just made me more suspicious. That’s why you hire a
risk officer and delegate! That’s the whole point! I could barely even hear Cohen blaming Caroline Ellison and Changpeng “CZ” Zhao for the debacle over the “no
risk officer” ringing in my ears.

Following the defense’s opening statements, things got still worse for Bankman-Fried. The prosecution called its first witness, Marc-Antoine Julliard, whose
money got stuck on FTX. Juilliard, who was born in Paris and lives in London, testified that he trusted FTX because Bankman-Fried came across as a leading
figure of the industry.
</item 13>

<item 14>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.
</item 14>

<item 15>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 15>

<item 16>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Witness Adam Yedidia answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in
New York City, October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Two of Sam Bankman-Fried's former friends from MIT, who also worked at crypto exchange FTX while living with the company's founder in the Bahamas, took the
stand in a Manhattan courtroom this week to testify against their former classmate, confidant, and boss — a man who allegedly ran a crypto empire that defrauded
thousands of customers out of billions of dollars. Gary Wang, the lesser-known co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday,
"Did you commit financial crimes while working at FTX?" "Yes," responded Wang.
</item 16>

<item 17>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Additionally, he was promised that Alameda had no preferential treatment on the FTX platform, even though the hedge fund was one of its top traders. Huang said
he was concerned about FTX's lack of a board of directors, but he eventually invested anyway. During cross-examination, Huang said Paradigm pressed
Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts. — CNBC's Dawn Giel
contributed to this report.
</item 17>

<item 18>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
That was all done by Bankman-Fried, he testified. In early 2020, Wang said he discovered for the first time Alameda's negative balance exceeded FTX's revenue,
an indication that Alameda was taking customer funds. Wang said he brought this to Bankman-Fried's attention several times. In late 2021, Wang discovered
Alameda had withdrawn $3 billion from its $65 billion line of credit. Wang's compensation was a base salary of $200,000 per year plus stock. He owned roughly
17% of FTX. Even though they were co-founders, "ultimately it was Sam's decision to make" when there were disagreements, he said.
</item 18>

<item 19>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 19>

<item 20>
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
</item 20>
=== END ITEM id=59881071d587554e ===

=== ITEM id=dcfba617831b210a ===
Gold sentences and facts:
Sentence 0: Weekly Breakdown MATCHUP @ ATL ATL -2.5 O/U 42.5 OPP VS WR 6th PROJ PTS 12.1 WR RNK 37th YTD Stats REC 25 TAR 31 REYDS 261 TD 1 FPTS/G 11.4 The only two games McLaurin posted good numbers in were Washington's 35-33 win over Denver in Week 2 and Washington's 34-31 OT loss to Philadelphia in Week 4.
  Fact 0.0: The matchup is at ATL, and the betting line is ATL -2.5.
  Fact 0.1: The over/under is 42.5.
  Fact 0.2: The opponent's ranking against wide receivers is 6th.
  Fact 0.3: The projected points are 12.1.
  Fact 0.4: The WR rank is 37th.
  Fact 0.5: Year-to-date receptions are 25.
  Fact 0.6: Year-to-date targets are 31.
  Fact 0.7: Year-to-date receiving yards are 261.
  Fact 0.8: Year-to-date touchdowns are 1.
  Fact 0.9: Fantasy points per game are 11.4.
  Fact 0.10: The only two games McLaurin posted good numbers in were Washington's 35-33 win over Denver in Week 2.
  Fact 0.11: The only two games McLaurin posted good numbers in were Washington's 34-31 overtime loss to Philadelphia in Week 4.
Sentence 1: Again, don't expect a massive game because the running game, defense, and special teams will have plenty of success in another big victory.
  Fact 1.0: The writer says readers should not expect a massive game.
  Fact 1.1: The running game will have plenty of success in another big victory.
  Fact 1.2: The defense will have plenty of success in another big victory.
  Fact 1.3: The special teams will have plenty of success in another big victory.

Retrieved items:
<item 1>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
PAST THREE WEEKS: McLaurin has led all Commanders wide receivers in targets and all Commanders players in receptions. He's the No. 1 guy, and his big Week 4
game (8-86-0 with an offensive fumble recovery for a touchdown) would have been even bigger if Sam Howell had better ball placement on a deep pass McLaurin was
open for.

BEARS: Are expected to play again without top outside cornerback Jaylon Johnson and safety Eddie Jackson. Not that those two absences led to a bonanza from
Broncos receivers; Russell Wilson spread the ball around. But if McLaurin remains the target leader then he'll have the best shot at a big game. Rookies Tyrique
Stevenson and Terell Smith figure to line up across from McLaurin the most.
</item 1>

<item 2>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Tack on the ascending secondary the Falcons seem to have and the Commanders' diffusion of targets and McLaurin is at best a low-end flex. I'd rather roll the
dice on Drake London, Tyler Lockett or Garrett Wilson.

LAST WEEK: Hopeful that his target edge over his teammates would keep growing, and even in a favorable game script where the Commanders were trailing by three
scores before halftime, McLaurin only saw five targets against the Bears' heavy zone coverage.

FILM: On the 10 third-down routes McLaurin ran last week, he drew extra coverage on four of them, got open on two of them (Howell threw elsewhere) and was a
decoy on Curtis Samuel's touchdown. What happened? Sam Howell was getting rid of the ball quickly and to shorter distances as the Bears were getting some
pressure on him. By the end of the game McLaurin's body language seemed dispirited.
</item 2>

<item 3>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Expect more of the same from QB Justin Herbert and his stud back against the Jets, who have allowed 46 receiving yards per game to RBs (sixth-most in the NFL).
They'll try to shut him down this Sunday, but it would feel so empty without AE!

Jahan Dotson, WR, Commanders @ Patriots — OVER 45.5 receiving yards

We've come full circle with Dotson, who made our preseason sleepers column then faded into early-season obscurity only to emerge again over the past two weeks.
Curtis Samuel was clearly not right last week, and the second-year wideout more than made up for it. Dotson caught 8-of-10 targets for 108 yards and a TD, a
stellar encore to his Week 7 line of five catches for 43 yards. The banged-up Patriots defense will be primarily focused on stopping stud Commanders receiver
Terry McLaurin, which should open up plenty of opportunities for Dotson.
</item 3>

<item 4>
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
</item 4>

<item 5>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sit Him (Lineup Decisions) Projections powered by Sportsline Terry McLaurin WR WAS Washington • #17

Age: 28 • Experience: 5 yrs. Weekly Breakdown MATCHUP @ ATL ATL -2.5 O/U 42.5 OPP VS WR 6th PROJ PTS 12.1 WR RNK 37th YTD Stats REC 25 TAR 31 REYDS 261 TD 1
FPTS/G 11.4 The only two games McLaurin posted good numbers in were Washington's 35-33 win over Denver in Week 2 and Washington's 34-31 OT loss to Philadelphia
in Week 4. This matchup has an over/under of 42.5 points, tied for fourth-lowest of the week.
</item 5>

<item 6>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Was the benefactor of Dak Prescott taking what the Patriots defense gave him, mostly utilizing Ferguson as a short-area target to fire the ball to
when the pressure was quick off the snap. But there were a couple of throws where Prescott read the defense well and found Ferguson for pretty good gains. All
told Ferguson led the Cowboys with seven targets and seven catches for 77 yards in three quarters of work.

THIS WEEK: The 49ers figure to unleash its pass rush on Prescott with zone coverage behind it, which only makes sense given how Prescott has handled the pass
rush pressure this year. It should mean more short throws for Ferguson.

49ERS: Are tops in football in catch rate allowed to tight ends (48.6%) and YAC/reception allowed (1.29 yards). They're also top-five in yards per catch (7.1).
</item 6>

<item 7>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five
catches, 157 yards, and two TDs. CeeDee Lamb (vs. Eagles) and the Cowboys look to avenge their road loss to the Eagles in Week 9, as Dallas' WR1 routinely has
double-digit reception potential.

Amon-Ra St. Brown (@ Bears) was relatively quiet this past week, but he's likely to rebound against a Bears defense that he totaled nine catches for 95 yards
against in Week 11. A.J. Brown (@ Cowboys) reached the century mark for the first time in four games in Week 13, and another high-usage day looks to be in store
for a potential shootout.

Keenan Allen (vs. Broncos) caught just five-of-nine targets last week, but as L.A.
</item 7>

<item 8>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP @ LAC DAL -2.5 O/U 51 OPP VS QB 21st PROJ PTS 18.9 QB RNK 15th YTD Stats PAYDS 1061 RUYDS 45 TD 5 INT 4 FPTS/G 13.8 Trusting Prescott
is a tall task, even in what looks like a favorable matchup. The Cowboys might pivot to a slower approach with Tony Pollard handling the run game more than he
has been over the past couple of games. That could take pressure off of Prescott, but it doesn't mean he'll unload statistically. I'd rather start Sam Howell,
Jared Goff and Matthew Stafford.

LAST WEEK: A nightmare scenario as Prescott tried to get aggressive on downfield throws early and missed until he hit KaVontae Turpin for a 26-yard touchdown.
</item 8>

<item 9>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Both scored.

BRONCOS: When McLaughlin, Williams and Perine were all part of the Broncos game plan in their Week 3 blowout loss, Williams had the most work (13 touches) and
the most PPR points (8.5). The only time a Denver back had 15-plus touches came in Week 1 when Williams had 17 against the Raiders (and had 9.7 PPR points, a
season-high).
</item 9>

<item 10>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
BILLS: Got their first test against tight ends last week in Evan Engram and swarmed him -- he caught 4 of 8 targets for 28 yards. He is used similarly to
Waller, which makes Buffalo's game plan for slowing Waller down easy to carry over this week. Their pass rush should also impact the Giants passing game and may
force Waller to run shorter routes so he gets open quicker.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sit Him (Lineup Decisions) Projections powered by Sportsline Dak Prescott QB DAL Dallas • #4

Age: 30 • Experience: 8 yrs.
</item 10>

<item 11>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
MATCHUPS: Each top-target earning wide receiver that plays mostly outside (like McLaurin) has found at least 10 PPR points in every game against the Bears. That
includes Rashee Rice in Week 3 (7 targets, 5-59-0), Courtland Sutton in Week 4 (5 targets, 3-27-1), Romeo Doubs in Week 1 (5 targets, 4-26-2) and Mike Evans in
Week 2 (8 targets, 6-171-1). Note that these receivers combined to catch 18 of 25 targets (72%) against Chicago.
</item 11>

<item 12>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
I'd expect that to be out the window now -- Flacco wasn't shy about chucking the rock deep with the Jets, attempting passes of 15-plus Air Yards on 20 percent
of his throws, and he had no problem focusing on one primary receiver (Garrett Wilson had 8, 14 and 11 targets in Flacco's starts). The hope is that both Flacco
and Cooper are experienced and capable enough to develop rhythm to salvage the Browns passing game. It's the only thing to buy into here as the Rams have
allowed the 12th fewest Fantasy points to wide receivers on the season and happen to have done well against deep throws against wide receivers (40% completion
rate allowed, two touchdowns, both top six on the season). Cooper's volume should hopefully push him to a decent Fantasy stat line. I'd give him a whirl over
Terry McLaurin, Marquise Brown and Diontae Johnson.
</item 12>

<item 13>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Easy peasy!

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Both teams had a week of rest, so don't treat this like a typical Thursday game. But do expect the Cowboys defense to keep the pressure on Geno Smith and a
Seattle offense that's struggled to gain consistency over the last month. Smith actually did OK completing passes last week when pressured (60% completion rate)
but not at an efficient level (5.4 yards per attempt). Just two QBs have exceeded 250 yards and four QBs have thrown multiple touchdowns against Dallas this
year; I don't like Smith's chances to do either, which hurts any upside case for any Seahawks receiver. By the way, Curtis Samuel was the first receiver to get
100 yards on the Cowboys this year, and it took him 12 targets and 9 catches to get there.
</item 13>

<item 14>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
O-LINE: It's possible that injuries that go as far back as to Week 2 along with some declining play from Tyler Biadasz and Terence Steele could be contributing
factors for Prescott's low ADOT. He doesn't want to get hit, so he's getting the ball out as fast as he can. Even against the 49ers he sped up his motion and
overshot those targets. He's just not as comfortable and the offense isn't adjusting well enough.

CHARGERS: Traditionally play a good amount of zone coverage but have changed it up more often this season and used more man-to-man, including versus Miami in
Week 1. But despite some headliners in the front seven and 16 sacks in their past three games, the Chargers rank fourth-worst in pass rush pressure rate at
29.5% this season. They also blitz at about the league average rate.
</item 14>

<item 15>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
McCaffrey, who seems poised to win the Offensive Player of the Year Award, leads the NFL in rushing (1,292 yards) and total yards per game (128.6) this season
and is tied for first in touchdowns (20).

Ravens TE Isaiah Likely thriving in the starting role: The Ravens offense hasn't lost a step since losing one of the team's top weapons in tight end Mark
Andrews (ankle) in Week 11, largely due to Likely's emergence.

The second-year tight end has averaged 64.3 yards over his last three games, which is particularly noteworthy since Andrews averaged 54.4 yards per game before
sustaining his injury.

Should we expect a defensive battle?: While both teams boast offenses that rank in the top five in yardage and scoring, this may not turn into a shootout.
</item 15>

<item 16>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
By the end of the game McLaurin's body language seemed dispirited.

COMMANDERS: Throw to their wideouts on 58.7% of their targets, which is below league average (but not as bad as, say, the Falcons at 36.4%). What makes that
number worse is that McLaurin has just one more target (31) than Jahan Dotson (30), who has three more targets than Curtis Samuel (27). They're all nearly even
after five games and each of them has had a turn leading in targets at least once.

FALCONS: Are expected to play a heavy dose of zone coverage, if only because the Commanders have struggled to be explosive against zone coverage this year.
Atlanta also gave a full dose of snaps last week to outside cornerback Jeff Okudah, who wasn't flawless but does offer another sizable body to help against
stronger perimeter receivers. Between Okudah and A.J.
</item 16>

<item 17>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
And truthfully, the only quarters they've actually played heavy zone have been at the end of blowout wins over the Giants and Jets. If they keep that up they'd
look completely different than in their playoff meeting against Purdy in January when they played heavy zone for nearly 86% of their snaps.

PURDY: Has been sensational against both man and zone coverages this season. His completion rate and yards per attempt are noticeably higher against zone, but
four of his five passing touchdowns have come versus man coverage. However, Purdy's been at his worst when pressured (like most quarterbacks), completing 48.6%
of his throws for 5.6 yards per attempt with a gaudy 18.9% off-target rate.

LAST YEAR IN THE PLAYOFFS: The Cowboys pressured Purdy on 48.5% of his drop backs despite blitzing just 21.2% of the time.
</item 17>

<item 18>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Between Okudah and A.J. Terrell there isn't a matchup that's great for McLaurin.

McLAURIN: Has an impressive 91.3% catch rate versus zone coverage this year, but is averaging just 9.3 yards per catch with it on a sub-9.0 ADOT. His target
share is below 20% regardless of the kind of coverage he sees.

Start Him (Lineup Decisions) Projections powered by Sportsline Logan Thomas TE WAS Washington • #82

Age: 32 • Experience: 9 yrs. Weekly Breakdown MATCHUP @ ATL ATL -2.5 O/U 42.5 OPP VS TE 31st PROJ PTS 8.4 TE RNK 11th YTD Stats REC 18 TAR 25 REYDS 183 TD 2
FPTS/G 11.6 Streaming Thomas is worth the risk considering how the Falcons have let up numbers to tight ends.
</item 18>

<item 19>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
PURDY: Has hit at least 20 Fantasy points in 8 of his 9 regular-season starts, and it really should be 9 for 9 because of that first touchdown that was ruled a
lateral back in Week 2. If you include the two playoff games he completed, Purdy has delivered at least 20 in 9 of 11 games. And in 5 of those 9 he hit at least
23 Fantasy points including each of his past two games.

COWBOYS: Dallas' defense lost top cornerback Trevon Diggs and they're still red hot. They lead the league in pass rush pressure rate this season (55.6%) and
they're in the top-10 in blitz rate (31.9%). It's that pressure that's helped the Cowboys rank in the top-five in completion rate allowed (57.4%), passing
touchdowns allowed (two) and missed tackles on pass plays (just five, that's best in football).
</item 19>

<item 20>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Bowden Jr., alongside A.T. Perry and Keith Kirkwood, are the only healthy Saints WRs right now, and a favorable matchup against Detroit gives each some flex
value in deep leagues. Mingo also draws a plus matchup against a reeling Buccaneers secondary, as a potential negative game script should boost his Week 13
fantasy ceiling.

Tillman is another talented rookie who's getting more run as the season progresses, and although P.J. Walker's struggled under center, Tillman is set up for a
high-usage day against the Rams with Amari Cooper (ribs) banged up.

WEEK 13 DFS:

DK lineup | FD lineup | Best stacks | Best values

Who are the biggest potential WR busts in Week 13?

Beware of Terry McLaurin (vs. Dolphins), Puka Nacua and Tutu Atwell (vs.
</item 20>

<item 21>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
On the flip side, three of the four quarterbacks Chicago faced found three touchdowns on their limited throws.

Start Him (Lineup Decisions) Projections powered by Sportsline Terry McLaurin WR WAS Washington • #17

Age: 28 • Experience: 5 yrs. Weekly Breakdown MATCHUP vs CHI WAS -6 O/U 44.5 OPP VS WR 19th PROJ PTS 15.2 WR RNK 21st YTD Stats REC 21 TAR 26 REYDS 212 TD 1
FPTS/G 12.1 The Commanders may wish to win this one with a big dose of the run game (Brian Robinson is a must-start), but if the Bears put up any offense at
all, McLaurin will be in position to put up nice numbers. He's a low-end WR2 worth the nod over George Pickens, Zay Flowers, Drake London and Jahan Dotson.
</item 21>

<item 22>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
They also blitz at about the league average rate. In theory this should not be a difficult matchup for Prescott, but

KELLEN MOORE: Prescott's QB coach and/or playcaller for the prior five seasons will be coaching the Chargers offense on Sunday, but he's sure to have spilled
the beans on Prescott's tendencies. We've seen offensive coordinators come back to wreck their old teams all the time and could see it again on Sunday.

FANTASY: The only quarterback to not get at least 22 Fantasy points against the Chargers was Raiders rookie Aidan O'Connell.
</item 22>

<item 23>
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
</item 23>

<item 24>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Plus the Browns just saw the Cowboys play 90% of their snaps in zone coverage and Purdy cooked them like BBQ chicken.

PURDY: Has been sensational against both man and zone coverages this season. His completion rate and yards per attempt are noticeably higher against zone, but
five of his seven passing touchdowns have come versus man coverage. However, Purdy's been at his worst when pressured (like most quarterbacks), completing 50%
of his throws for 6.7 yards per attempt with a gaudy 15.9% off-target rate. These stats are actually better than they were last week.

BROWNS: Rank third in pass rush pressure rate at 41.5% and blitz at an above-average 29.3% rate. Granted, the Cowboys ranked higher than that last week and got
beat anyway.
</item 24>

<item 25>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
In their past three games, they gave up 4.1 yards per rush to the Ravens (just 6.3% of their runs went 10-plus yards), 7.8 yards per rush to the Steelers thanks
to a long TD run by Jaylen Warren (23.8% of their runs went 10-plus yards) and 4.6 yards per rush to the Broncos (18.5% of their runs went 10-plus yards).
Playing more snaps because the Browns offense has been inept isn't the reason for it either because they struggled with the run in the first half against all
three teams (over 5.3 yards per carry allowed against each). It's silly to say the Rams passing game is struggling when Matthew Stafford just threw four
touchdowns, but Cooper Kupp looked like he played hurt last week and Puka Nacua didn't look that much better.
</item 25>

<item 26>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ DAL DAL -9 O/U 47.5 OPP VS WR 17th PROJ PTS 11.1 WR RNK 22nd YTD Stats REC 43 TAR 80 REYDS 678 TD 3 FPTS/G 12.9 Metcalf has at least
nine targets in each of his past three games and in five of his past six. That kind of usage, especially in a game where the Seahawks figure to pass a bunch,
makes him a borderline must-start. But on the 12 targets he's seen that have traveled at least 15 Air Yards over the past four weeks, Metcalf has caught just
three. More alarming: Five of the targets were knocked away by defenders who were right on him when the ball arrived. He's not really zooming past anyone these
days, and he's not winning contested catches regularly either.
</item 26>

<item 27>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Dolphins), Puka Nacua and Tutu Atwell (vs. Browns), Drake London (@ Jets), and Tyler Lockett and Jaxon Smith-Njigba (@ Cowboys) this week.

McLaurin's averaging just 42 receiving yards over his past three games, and a heavy dosage of Jalen Ramsey hinders his Week 13 fantasy value.

Nacua and Atwell see a Cleveland coverage unit who entered Week 12 allowing the fewest fantasy points per game to WRs. Heck, even Cooper Kupp, who hasn't looked
like himself of late, could underperform again.

We can't trust London and the rest of the Falcons passing offense against a well-rounded Jets defense. To make matters worse, he'll be matched up against Sauce
Gardner for most of the day, too.

Lockett and Smith-Njigba might be better suited for the bench in PPR leagues against a lockdown Dallas coverage unit.
</item 27>

<item 28>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ DAL DAL -9 O/U 47.5 OPP VS WR 10th PROJ PTS 10.6 WR RNK 31st YTD Stats REC 54 TAR 79 REYDS 575 TD 4 FPTS/G 12.5 Lockett has
seven-plus targets in four of his past five, but he still doesn't have as many as Metcalf's gotten in those games. He also doesn't see nearly as many deep
throws as Metcalf -- literally two targets in his past two games traveled 15 or more Air Yards. Obviously he has potential to always pop off for a big game like
he did against a tough Browns defense five weeks ago, but recent matchups against the Niners, Rams and Ravens weren't kind to Lockett. They were even less kind
to Geno Smith, and this week shouldn't be much different.
</item 28>

<item 29>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Only Joshua Dobbs, who ran for 55 yards, has scored double-digit Fantasy points against the Niners (18 points). Not that the Cowboys have been tested much --
Daniel Jones, Zach Wilson and the Patriots QBs are the other guys they've man-handled.

O-LINE: San Francisco has allowed a pass rush pressure on 36.4% of their snaps, which sounds like a lot but it's just a shade worse than league average. That
number is weighted by the Giants' decision to blitz on 84.6% of Purdy's dropbacks in Week 2, creating an inflated pass rush pressure rate. Despite the Giants'
efforts, Purdy still threw for 310 yards and two touchdowns in a game he didn't have Brandon Aiyuk in.

COWBOYS: Have become one of the most aggressive defenses in the league, ranking third from the bottom in zone coverage snaps.
</item 29>

<item 30>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
The Cowboys wound up paying the price through the air for playing a ton of zone coverage instead. It was suggested by a friend that they didn't want to get
burned by long runs by Christian McCaffrey. I guess that worked. The combination of the 49ers defense running hot and the Browns offense figuring to get stuffed
should make for at least a game script for Purdy to have decent numbers. I'm not making the mistake again of benching Purdy -- I'd start him over Trevor
Lawrence, Geno Smith and Jared Goff.

LAST WEEK: Took on one of the toughest pass rushes in football and slayed them for four touchdowns on just 24 pass attempts. Of his seven incompletions, one was
dropped, two were batted at the line of scrimmage, another was a throwaway and two more were defensed. Purdy was off-target on 0.0% of his throws.
</item 30>
=== END ITEM id=dcfba617831b210a ===
