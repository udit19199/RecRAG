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

=== ITEM id=6cff4187eea73554 ===
Gold sentences and facts:
Sentence 0: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 0.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 0.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 0.2: The case argues that Google siphons off news publishers' readers.
  Fact 0.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 0.4: The case argues that Google does this through anticompetitive means.
Sentence 1: Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part.
  Fact 1.0: Wired removed an article for not meeting its standards.
  Fact 1.1: The article was shared widely on social media.
  Fact 1.2: The article was boosted in write-ups at other outlets.
  Fact 1.3: The degree to which the article was shared and boosted shows how willing people are to accept foul play on Google's part.

Retrieved items:
<item 1>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 6>

<item 7>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 7>

<item 8>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 10>

<item 11>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 11>

<item 12>
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
</item 12>

<item 13>
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
</item 13>

<item 14>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 14>

<item 15>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
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
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 17>

<item 18>
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
</item 18>

<item 19>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 19>

<item 20>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 20>
=== END ITEM id=6cff4187eea73554 ===

=== ITEM id=b757b0275b139c9f ===
Gold sentences and facts:
Sentence 0: At the time, the music video for the song – which featured a woman with blonde hair - sparked rumours that he and Spears broke up because she allegedly cheated on him.
  Fact 0.0: At the time, the music video for the song featured a woman with blonde hair.
  Fact 0.1: The music video sparked rumours that he and Spears broke up.
  Fact 0.2: The rumours said the breakup happened because Spears allegedly cheated on him.
Sentence 1: Also, another thing that hints towards their breakup is Danis’s first press conference for the Logan Paul fight.
  Fact 1.0: Danis’s first press conference for the Logan Paul fight hints towards the breakup.
  Fact 1.1: Danis held a first press conference for the Logan Paul fight.

Retrieved items:
<item 1>
Truth Behind Jake Paul Dating Dillon Danis’ Ex-girlfriend: Did MMA Star Break Up With Savannah Montano? (Essentially Sports, 2023-10-07T21:14:09+00:00)
Dillon Danis has been going back and forth with the Paul family long before his upcoming fight against Logan Paul, scheduled for October 14th. At the moment,
Danis is preparing hard for his Misfits contest, as he’s tried to get in the head of his opponent leading up to the bout. As a result, Danis has garnered over a
million followers on X. ‘El Jefe’ has been trying to dig up the past relationships of Logan Paul’s fiancée, Nina Agdal, as a way to create turmoil in Paul’s
mindset during the fight.

America’s Favorite Video Today

However, this beef was first started by Logan Paul’s younger brother, Jake Paul. He called out Danis on his Instagram story in 2020, by taking a hit at his
rumored ex-girlfriend, Savannah Montano. So Danis’s target at Agdal could have a revenge angle that many didn’t know about.
</item 1>

<item 2>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
“Whether he realises it or not, Mr Federline has not only violated the privacy and dignity of the mother of his children, he has undermined his own children,
whose privacy he should protect,” the statement read. “Putting aside his ITV interview, Mr Federline’s ill-advised decision to post an old video of his 11 and
12-year-old children was cruel, bottom of the barrel stuff. It was abhorrent.”

“In addition to demeaning himself and violating societal norms, he has now also created various legal issues for himself, including, but not limited to,
implicating cyber-harassment and cyber-bullying statutes, among other things.”

The Woman in Me hits shelves in the US and UK on 24 October.
</item 2>

<item 3>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
While celebrities like Madonna, Paris Hilton, and Selena Gomez attended the festivities, Spears’ close family – her mother Lynne, father Jamie, sister
Jamie-Lynn, and her two sons – were not present at the nuptials.

Although their marriage last only 14 months, with Asghari filing for divorce in August this year, the actor and former dancer was previously praised by fans
over his continuing support for Spears amid her conservatorship and custody battle.

In August 2022, Federline claimed during an interview with ITV that their teenage sons found it “tough” seeing their mother sharing nude photos online, saying:
“I can’t imagine how it feels to be a teenager having to go to high school.”

Spears hit back at her ex-husband in a statement shared to her Instagram Stories responding to Federline’s “hurtful” comments.
</item 3>

<item 4>
Truth Behind Jake Paul Dating Dillon Danis’ Ex-girlfriend: Did MMA Star Break Up With Savannah Montano? (Essentially Sports, 2023-10-07T21:14:09+00:00)
Savannah, how you doing? What’s good?” However, Montano completely denied any relation with the Influencer boxer whatsoever. She responded to Paul’s stories
through a video and gave her side of the story.

She said, “I’m dead confused at some of you guys going, ‘Oh, but why did you answer his call?’ As if I have his f**king number saved in my phone! It was
literally a random number. You can literally see in the video, I’m like, ‘Who is this?’” So she denied the accusations of hooking up with Paul and even felt
disgusted by the speculations.

However, her current relationship status is unknown. Even three years later, many followers are still not convinced about her breakup with Dillon Danis.

Dillon Danis and Savannah Montano breakup reality

As of now, it’s not completely clear whether the couple is still together or not.
</item 4>

<item 5>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Timberlake proves to be the last person you’d call in a crisis. “At some point he thought maybe music would help, so he got his guitar and he lay there with me,
strumming it,” she writes. Almost as embarrassing is husband Kevin Federline’s attempt to start a music career. “He really thought he was a rapper now,” she
writes, with the bafflement of someone who has never known pretension. “Bless his heart – because he did take it so seriously.”

Motherhood leaves her vulnerable

When Britney gets pregnant with her first son, she initially thinks it might protect her: “I wanted everyone to stay away: stand back! There’s a baby here!”
Inevitably, it doesn’t work out that way. The paparazzi become aggressive, particularly once she has her kids in quick succession – becoming pregnant a second
time three months after giving birth to her first.
</item 5>

<item 6>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Jason Trawick: 2009 to 2013

(Getty Images)

Spears and Jason Trawick, a Hollywood agent who worked at WME, started dating in 2009 before getting engaged in December 2011. However, they called off their
engagement in January 2013 and announced they were going their separate ways.

"Jason and I have decided to call off our engagement. I’ll always adore him and we will remain great friends," Spears said in a statement sent by her
representative to ABC News.

Trawick also issued a statement at the time, saying: “As this chapter ends for us a new one begins. I love and cherish her and her boys and we will be close
forever."

David Lucado: 2013 to 2014

Two months after ending her engagement with Trawick, Spears started dating David Lucado, who worked at a legal firm. In August 2014, Spears announced their
split, tweeting: “Ahhhh the single life!”
</item 6>

<item 7>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Outsiders are now aware that Jada and Will have been separated since 2016, but the timeline of her relationship with Alsina is still fuzzy. Indeed, the terms of
Will and Jada’s separation are still unknown to the public and in some cases, having separate relationships can give couples the clarity they need about their
marriage.

“What’s pivotal here is, what was the agreement? Did we agree to separate to work on ourselves for the purpose of coming back together again?” Winter said. “Did
we agree to separate and just be alone for a while and not have that discussion? And in the second case, if you have an affair, sometimes it can bring you back
to the original person. You can get whatever you need to achieve or gain a different perspective, allowing you to decide if you want to come back or leave for
good.”
</item 7>

<item 8>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
After months of court discussions, Federline rejected Spears’s child support proposal and instead doubled his request to $40k a month. A settlement was finally
reached between the two parties in September 2018, with Spears increasing her child support payments to Federline from $20,000 to $35,000 a month.

Amid her father’s health crisis and ongoing emotional distress, Spears voluntarily checked into a mental health treatment facility in 2019. After hearing the
news, Federline’s attorney Mark Kaplan told Entertainment Tonight: “Kevin commends Britney for recognising that she needs to take a step back and that she is
taking the reasonable steps to dealing with her situation in a responsible way. He wishes everyone well.” That same year, Federline was granted 70 percent
custodial rights and Spears 30 per cent.

After years with her father in control of her finances and personal life, Spears was finally granted freedom from her conservatorship in November 2021.
</item 8>

<item 9>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 9>

<item 10>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Billionaire entrepreneur Mark Cuban, who paid $1.8 million (approx.) in penalties on behalf of his team, the Dallas Mavericks, fondly remembered the former NBA
commissioner David Stern, on his death in January 2020. Stern came and instituted an anti-drug agreement and even banned All-Star player Michal Richardson for
life for drug abuse. The revitalized image of the NBA helped rake in the moolah from sponsors and made it more appealing to fans. He even went against the tide
to open up the Olympic basketball competition to NBA players resulting in the famous “Dream Team” of 1992. America has never failed to win a medal in the 19
Olympic basketball games that it has played since then.

ADVERTISEMENT Article continues below this ad

The NFL has had its fair share of, um, let’s say, “bumps” down the road as well.
</item 10>

<item 11>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
After her split from Justin Timberlake, she dreams about quitting pop and opening a shop: “With the gift of hindsight I can see that I hadn’t given myself
enough time to heal from the breakup,” she writes – though her requests for a break are not only turned down, but interpreted as a potential sign of sedition to
be quashed and interrogated.

In this restrictive environment, small kindnesses take on an outsized significance, like Paris Hilton encouraging her “to have fun for the first time in a long
time”, leading to some infamous nights out with the heiress and Lindsay Lohan. Once again, Britney is seen as stepping out of bounds, potentially damaging her
earning potential. One night she has her mother babysit her sons; when she returns home tipsy, Lynne Spears screams at her, she writes. “The shame I felt killed
my heart. I stood there, reeling, and thought, OK, I guess it’s forbidden for me to party.”
</item 11>

<item 12>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
In interviews, Taylor delighted in revealing this was based on a true story.

"It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something
really major and traumatic happen to me and they don't know about it," she told NPR.

In her Rolling Stone cover story, Taylor said the crash actually occurred on a snowmobile "with an ex who lost control and wrecked it so badly that she saw her
life flash before her eyes".

That prompted speculation the track was also about Styles: The couple had gone skiing with Justin Bieber and Selena Gomez in late 2012.

The lyrics also reference their identical aeroplane necklaces and a December break-up, which ties in with the "Sad Taylor on a boat" meme.

But the singer steadfastly refuses to name names: "The fact that I've never confirmed who those songs are about makes me feel like there is still one card I'm
holding," she told GQ.
</item 12>

<item 13>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
At the time, her representative also confirmed to People: “Yes, she’s single.”

The news of the breakup came asTMZ alleged that Lucado had cheated on Spears, with claims that her father had purchased a video of the infidelity incident. The
publication later alleged that Lucado denied the cheating allegations, with claims that he was seeing the other woman while he and Spears were broken up.

Despite their seemingly rocky history, Lucado showed his support for his ex in a 2016 interview with theNew York Times about her conservatorship. Speaking to
the publication, he said: “If anyone knew the real Britney, they would know that she would rather be remembered for being the great mother she is rather than
the artist she is.”

The publication also noted that Lucado denied the infidelity claims from Spears’ father, before he added: “And if anyone could see her interactions with her
kids, they would know that there is no need for a conservatorship over Britney’s personal life.”
</item 13>

<item 14>
Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo (FOX News - Entertainment, 2023-12-17T10:00:59+00:00)
You just get this profound feeling of instability." He stopped drinking altogether in 2013.

Micheál and Daniel, now 28 and 27, are both actors. Micheál took on his mother's surname of Richardson, something Neeson said was "the right thing" to do.

"I think it was a lovely homage, a nice gesture and he's, you know, he's not saddled with my last name which, I have a certain celebrity status I guess," Neeson
told Andy Cohen in 2020. "I'd hate for him to be constantly asked, 'Oh, are you Liam Neeson's son?' So, it was a lovely gesture. Natasha's family, mother and
sisters, were very touched by it, as indeed I was, too."

Kevin Federline

Kevin Federline met Britney Spears in 2004 at a nightclub, and months later they were married.
</item 14>

<item 15>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Spears is also the mother of two children – Sean, 18, and Jayden, 17 – who she shares with her second ex-husband, Kevin Federline.

According to a press release for The Woman in Me, Spears’ memoir is “a brave and astonishingly moving story about freedom, fame, motherhood, survival, faith,
and hope”. Spears’ book is also expected to cover the high-profile legal battle to end her conservatorship, which was dissolved in November 2021 after 13 years.

Throughout her time in the spotlight, specifically during the ‘90s and 2000s, Spears has been in some heavily publicised romances. Unsurprisingly, fans are
anticipating that Spears will detail some of her relationships in The Woman in Me, which is set for release on 24 October.

From Justin Timberlake to her now ex-husband Sam Asghari, here’s everything we know about Britney Spears’ relationships and rumoured romances.
</item 15>

<item 16>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
“I learned real fast how much of a whirlwind the press and everything was.”

After just three months of dating, the two tied the knot in September 2004 during a surprise wedding ceremony in Studio City, California. Federline was 26 years
old at the time, while Spears was just 22. Earlier that year, Spears was also married to her childhood friend Jason Alexander, but their marriage was annulled
after just 55 hours.

The former couple welcomed their first child, son Sean Preston, weeks before their one-year wedding anniversary, and their second son Jayden James in 2006. At
the time, Federline was already a parent to his daughter Kori Madison, now 21, and son Kaleb Michael, 19, with ex fiancée Shar Jackson.

Just eight weeks after the birth of their second child, Spears filed for divorce from Federline in November 2006, citing “irreconcilable differences”.
</item 16>

<item 17>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
He later issued an apology to Spears on Instagram, writing: “I’ve seen the messages, tags, comments, and concerns and I want to respond. I am deeply sorry for
the times in my life where my actions contributed to the problem, where I spoke out of turn, or did not speak up for what was right. I understand that I fell
short in these moments and in many others and benefited from a system that condones misogyny and racism.”

In June 2021, he also shared his support for Spears after she testified publicly in a bid to end her conservatorship, which was controlled by her father, Jamie
Spears. In a Los Angeles court hearing at the time, Spears alleged that she had been forced to take lithium against her will. She also claimed that the team
managing her conservatorship had prevented her from having another baby by refusing to remove her IUD.

“After what we saw today, we should all be supporting Britney at this time.
</item 17>

<item 18>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Justin Timberlake: 1998 to 2002

(Getty Images)

Spears’ first highly public relationship was with her co-star in the Mickey Mouse Club, Justin Timberlake. The pair made many appearances together while dating,
including the iconic moment Spears and Timberlake wore matching, all-denim outfits to the American Music Awards in 2001.

They broke up in 2002, shortly after Timberlake and his boy band NSYNC released their hit tune, “Cry Me a River”. At the time, the music video for the song –
which featured a woman with blonde hair - sparked rumours that he and Spears broke up because she allegedly cheated on him.

Although the former couple never acknowledged the breakup rumours, the 2021 Framing Britney documentary addressed some of the public scrutiny that Spears faced
during the 2000s. Following the documentary’s release, Timberlake came under criticism for the way he acted following his breakup with Spears.
</item 18>

<item 19>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
The singer made a powerful testimony at her conservatorship hearing that June, where she spoke out for the first time about her experience, calling for those
involved to be “put in jail”. Following the testimony, Federline expressed his support for the singer, saying he “only hopes the best for her because when the
best for her is achieved, it’s the best for their kids.”

“The best thing would be for their mom to be healthy and happy,” his lawyer said in a statement at the time. “And if either of those things aren’t true, it
doesn’t provide for the best setting for custody to be exercised.”

Britney Spears poses with her sons Jayden James and Preston at Dodger Stadium in 2013 (Getty Images)

In June 2022, Spears married her longtime boyfriend Sam Asghari during an intimate ceremony at their Los Angeles home, just one month after the singer revealed
she suffered a miscarriage.
</item 19>

<item 20>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
“It saddens me to hear that my ex husband has decided to discuss the relationship between me and my children … As we all know, raising teenage boys is never
easy for anyone,” she wrote. “It concerns me the fact that the reason is based on my Instagram … It was LONG before Instagram … I gave them everything … Only
one word: HURTFUL.”

Asghari went on to accuse Federline of using their sons to “vilify” his wife, and claimed her ex-husband’s “gravy train will end soon, which probably explains
the timing of these hurtful statements.”

“There is no validity to his statement regarding the kids distancing themselves and it is irresponsible to make that statement publicly,” Asghari wrote on
Instagram.

Following Federline’s comments, Spears’s attorney Mathew Rosengart said in a statement obtained by Variety that Federline has created “legal issues” for
himself, after posting private videos of Spears sternly speaking with her children without her consent.
</item 20>

<item 21>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Not to mention her rumoured relationship with Kansas City Chiefs tight end Travis Kelce caused a 400 per cent spike in NFL jersey sales.

Since her first NFL appearance in September, Swift’s occasional attendance at Kelce’s games has received pushback from NFL fans who believe her presence is
distracting from the game itself. “I’m just there to support Travis,” Swift admitted. “I have no awareness of if I’m being shown too much and pissing off a few
dads, Brads, and Chads.”

She playfully added: “Football is awesome, it turns out. I’ve been missing out my whole life.”

The “Bad Blood” singer revealed that their romance began when Kelce gave her shoutout on his podcast, admitting he attempted to give Swift his phone number
after one of her Eras Tour shows. The pair immediately began “hanging out” and by the time she attended her first Chiefs game in September, they were already a
couple. “I think some people think that they saw our first date at that game?
</item 21>

<item 22>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Charlie Ebersol: 2014 to 2015

Spears and Charlie Ebersol, who is known as the co-founder of Tempus Ex Machina and The Company, started dating in 2014. While their relationship was fairly low
profile, Spears previously opened up about why she was drawn to Ebersol in the first place.

“There was something about him that made me feel really peaceful and at ease. And he was really hot, too. Adorably hot,” she told People at the time.

They officially broke up in June 2015 after eight months of dating.

Sam Asghari: 2016 to 2023

(Getty Images)

The pair first met in 2016 when Asghari was cast as Spears’ love interest in the music video for her song, “Slumber Party”, from her 2016 album Glory.
</item 22>

<item 23>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
They now share two daughters, Jordan, 10, and Peyton, eight.

Adnan Ghalib: 2007 to 2008

Spears met British paparazzo Adnan Ghalib in 2007, shortly after she decided to shave her hair off in front of approximately 70 photographers. They only dated
for about a year, as the relationship ended in 2008.

In the 2021 Netflix documentary, Britney vs Spears, Ghalib spoke candidly about the romance, claiming: “It was a very simple, very easy relationship, you know?
But between her fans and everybody else, it was like I was the bad guy.”

The former couple split in 2008 after Ghalib reportedly pulled a pregnancy test-buying stunt for publicity. In 2009, Jamie Spears filed a restraining order
against Ghalib, which ordered the photographer to stay away from the singer and her family until 2012.
</item 23>

<item 24>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
Spears has previously been open about her co-parenting struggles with Federline. Last year, he claimed their children were keeping their distance from Spears,
who was embracing her newfound freedom following the end of her 13-year long conservatorship by posting several nude images on social media.

From marriage to divorce to co-parenting, here is a complete timeline of the drama between Britney Spears and Kevin Federline.

The “Toxic” singer and the former backup dancer began their whirlwind romance back in 2004, when Federline worked as a backup dancer for Justin Timberlake.
Federline recalled his first meeting with Spears to People back in 2008.

“I met her at a club in Hollywood, Joseph’s. Our eyes met and that was it. We just hit it off right away,” he said. “I learned real fast how much of a whirlwind
the press and everything was.”
</item 24>

<item 25>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
“After what we saw today, we should all be supporting Britney at this time. Regardless of our past, good and bad, and no matter how long ago it was,” Timberlake
wrote on X, formerly known as Twitter. “What’s happening to her is just not right. No woman should ever be restricted from making decisions about her own body.”

Wade Robson: 2001 to 2002

Shortly after her breakup from Timberlake, Spears was rumoured to be dating Wade Robson, a dancer and choreographer who appeared in the music video for her 2009
tune, “(You Drive Me) Crazy”. However, neither Spears nor Robson confirmed these rumours.

Colin Farrell: 2003

(Getty Images)

Back in 2003, Farrell and Spears made headlines when they were spotted holding hands at the premiere of The Recruit, sparking speculation that they were dating.
</item 25>

<item 26>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Justin had embarrassed me publicly, so my rebuttal onstage was to kind of go there a little bit, too. But it was absolutely horrible.”

She has an immaculate sense of shade

Britney seems too pure of heart for active malice, but she has an immaculate way with a piercing characterisation. Timberlake’s band, ‘NSync, were big hip-hop
heads, she writes, and sometimes “tried too hard to fit in” with Black artists. When Timberlake spots Ginuwine at an event, he “got all excited and said, so
loud, ‘Oh yeah fo shiz, fo shiz! Ginuwiiiine! What’s up, homie!’”

Her account of being encouraged to abort their pregnancy at home, lest anyone see them going in and out of hospital, is horrifying. Britney, with no pain
relief, is writhing on the floor. Timberlake proves to be the last person you’d call in a crisis.
</item 26>

<item 27>
Truth Behind Jake Paul Dating Dillon Danis’ Ex-girlfriend: Did MMA Star Break Up With Savannah Montano? (Essentially Sports, 2023-10-07T21:14:09+00:00)
Instead, he ended up taking the fight against the former Olympic MMA artist, Ben Askren in 2021. He again won the fight by a knockout, while many fans still
thought that Danis was smart to not respond to the Youtuber-turned-boxer. Still, Paul tried his best to involve Montano in their non-existent tussle.

via Getty TAMPA, FLORIDA – DECEMBER 18: Jake Paul fights Tyron Woddley during a rematch of an eight-round cruiserweight bout at the Amalie Arena on December 18,
2021 in Tampa, Florida. (Photo by Mike Ehrmann/Getty Images)

Besides the Instagram Stories, Jake Paul also told TMZ, about the entanglement with Savannah Montano. He said, “Dillon, I know you’re watching this … Just
remember you’re still a bitch and I hooked up with your girlfriend. Savannah, how you doing? What’s good?”
</item 27>

<item 28>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
“I’m her first husband, I’m here to crash the wedding.”

After he was arrested for trespassing, vandalism, and battery, a security guard later revealed in court that when Alexander crashed the wedding, he also tried
to enter Spears’ bedroom.

Kevin Federline: 2004 to 2007

(Getty Images)

Kevin Federline and the “Gimme More” singer first met at a club in 2004, before they went on to start their relationship. After just three months of dating, the
two tied the knot in September of that year during a surprise wedding ceremony. Weeks before their one-year wedding anniversary, they welcomed their first
child, son Sean Preston.

In 2006, they welcomed their second son, Jayden James. At the time, Federline was already a father to his daughter Kori Madison, now 20, and son Kaleb Michael,
18, with ex-fiancée Shar Jackson.
</item 28>

<item 29>
Truth Behind Jake Paul Dating Dillon Danis’ Ex-girlfriend: Did MMA Star Break Up With Savannah Montano? (Essentially Sports, 2023-10-07T21:14:09+00:00)
Even Logan addressed the topic of Montano and Jake during their first press conference. So, who is this girl? What’s the actual story here?

Who is Dillon Danis’ ex-girlfriend often linked with Jake Paul?

ADVERTISEMENT Article continues below this ad

Savannah Montano was quite a trending name back in 2020, and that was mostly because of her relationship with the Bellator MMA fighter, Dillon Danis. Montano is
a successful Instagram model, with around 1.6 million followers on her social media. The couple turned a lot of heads during their relationship, sharing some
NSFW steamy pictures and videos for their fans.

Despite their controversial relationship, Montano also gained some attention after Jake Paul used her to call out Dillon Danis for a fight. After knocking out
the ex-NBA star, Nate Robinson, ‘The Problem Child’, was on the hunt to find his next opponent.
</item 29>

<item 30>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
$28 billion. And just like that, their target will be met.

USA Today via Reuters NFL, American Football – Taylor Swift reacts while sitting next to Donna Kelce watching the Kansas City Chiefs vs Chicago Bears game
during the first half at GEHA Field at Arrowhead Stadium, Kansas City, Missouri, USA | Image Credits: Denny Medley-USA TODAY Sports/Reuters

And while analysts like Stephen A. Smith may go around telling people they should be embarrassed of themselves for getting so invested in someone else’s
relationship, it is good business for the NFL. TMZ reported that #87’s jersey sales skyrocketed 400% and his Instagram followers by 276%. Yahoo Finance reported
a 53% rise in viewership among teen girls. Who would have thought in their wildest dreams that the NFL would manage to rope in this part of the population? The
October 1 matchup between the Chiefs and the Jets drew 27 million viewers.
</item 30>

<item 31>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

In her forthcoming memoir, The Woman in Me, Britney Spears is expected to reflect on some of the most pivotal moments in her life, including her relationships.

Over the years, the pop star’s love life has continued to make headlines. Most recently, her ex-husband Sam Asghari filed for divorce after only 14 months of
marriage.
</item 31>

<item 32>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
“It’s an LAT partnership, meaning ‘living apart together,’” Winter said. “While everybody is at a different point as to why they are doing this, the centre
point is the same: the preservation of the partnership is paramount. Separations can also be beneficial if both individuals are working on themselves and have
the mutual goal of reuniting, or making their relationship better. However, that doesn’t necessarily mean that their improved relationship will result in them
living together again.”

On the other hand, dating coach Jacob Lucas - who’s based in the UK - acknowledged that not all separations end in resolutions. When couples opt not to live
together anymore, it could create an emotional disconnect between them. If a lack of connection occurred before couples separated, that separation doesn’t
necessarily make the issues go away. In fact, Lucas believes that in these situations, couples could also be avoiding the inevitable breakup for the sake of
comfortability.
</item 32>

<item 33>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Revisiting the NBA caricature back in time, the New York Knicks became the first all-Black team in 1979. But their fans made their disapproval very apparent.
“It wasn’t so much New York attitudes, but fans around the league let you know how they felt about you as an opponent and as a Black man as well,” Knicks star
Earl Monroe told Newsday.

USA Today via Reuters May 15, 2021; San Antonio, Texas, USA; Phoenix Suns forward Torrey Craig (12) and San Antonio Spurs forward Keldon Johnson (3) fight for
loose ball as Suns guards Devin Booker (1) and Chris Paul (3) look on in the third quarterat AT&T Center. Mandatory Credit: Scott Wachter-USA TODAY Sports

We could list out all the scandals, fights, and controversies here, but you get the idea.
</item 33>

<item 34>
Truth Behind Jake Paul Dating Dillon Danis’ Ex-girlfriend: Did MMA Star Break Up With Savannah Montano? (Essentially Sports, 2023-10-07T21:14:09+00:00)
At the time, the name that popped up the most was 2-0 Dillon Danis. So, when he shared the screenshot of the FaceTime with Danis’s ex-girlfriend, the fans
started drawing connections.

Paul somehow found the model’s number and sent her a FaceTime request. Many started to believe that Montano was having an affair and cheating on her
then-boyfriend Danis. Paul posted two stories. In the first one, he wrote, “Aye [Dillon Danis], come get your girl back.” And then in the second one, he took
another hit at the fighter by writing, “Can’t wait to see you soon.” So this begs the question, is it true? Was Paul actually involved in an affair with
Savannah Montano?

Did Jake Paul and Savannah Montano hook up?

This callout didn’t help much, as Jake Paul never could finalize his fight with Dillon Danis.
</item 34>

<item 35>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
They made their relationship Instagram official in 2017 and went on to support each other through different milestones - including the 29th annual GLAAD Media
Awards, where Spears accepted the Vanguard Award for her work promoting equality and acceptance.

In April 2022, Spears announced she was pregnant with her and Asghari’s first child together, and her third child. However, the following month, she said in a
statement posted to Instagram that she had suffered a miscarriage. The couple wrote: “We have lost our miracle baby.”

That June, they held their lavish nuptials at her home in California with a star-studded guest list, which included Selena Gomez, Paris Hilton, Drew Barrymore,
and Donatella Versace - who designed Spears’ wedding dress.

In August, months after Asghari shut down speculation about getting a divorce, TMZ claimed that he had confronted Spears about rumours she cheated on him.
</item 35>

<item 36>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.
</item 36>

<item 37>
Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo (FOX News - Entertainment, 2023-12-17T10:00:59+00:00)
They welcomed two sons together, Preston and Jayden, before divorcing in 2007. At the time of the divorce, they split custody, but a few months later, Federline
would receive sole physical custody because of what a judge called Spears' "habitual, frequent and continuous use of controlled substances and alcohol."

In 2008, when Spears began experiencing trouble with her mental health, Federline was awarded sole legal custody as well. Spears was given visitation rights. In
her recently released memoir, "The Woman in Me," she claimed that she agreed to her conservatorship for her kids.

"Because I played by the rules, I was reunited with my boys," Spears explained.

She eventually gained more custody, though Federline has remained the kids' primary caregiver. They have lived with him since the divorce. Earlier this year,
they moved to Hawaii with Spears' approval.

After years of getting along, some tension began to show in 2022.
</item 37>

<item 38>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
However, during an interview with Entertainment Tonight at the time, the actor specified that he and the singer were just friends.

“She’s just a mate,” Farrell told the outlet, as reported by People in 2003. “Seriously, we just met a week ago, [we’re] having a laugh.”

Jason Alexander: 2004

Jason Alexander was Spears’ first husband, tying the knot at a wedding chapel in Las Vegas in 2004. However, only 55 hours after the nuptials, they annulled the
marriage.

In 2022, Alexander made headlines when he tried to attend Spears’ wedding to Asghari, as could be heard saying in a video that he was going to crash the
nuptials. “She’s my first wife, my only wife,” Alexander reportedly told security. “I’m her first husband, I’m here to crash the wedding.”
</item 38>

<item 39>
Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo (FOX News - Entertainment, 2023-12-17T10:00:59+00:00)
After years of getting along, some tension began to show in 2022. Spears and Federline both shared negative posts on social media about the other, which they
both deleted. Soon after, Preston and Jayden did a rare interview with ITV to discuss the situation.

While Jayden clarified that there was "no hate" involved towards anyone, he said that "it will take a lot of time and effort" to fix their relationship with
their mother. He clarified, "I 100% think this can be fixed. It's just going to take a lot of time and effort. I just want her to get better mentally. When she
gets better, I really want to see her again."

In a message directly to Spears, he said, "I love you a lot, I hope for the best for you. Maybe one day we can sit down like this and talk again."

LIKE WHAT YOU’RE READING?
</item 39>

<item 40>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
“I already read it and I’m very proud of her,” he told the outlet. “She put a lot of work into it, and it was very hard. I’m definitely excited, and I’m
extremely proud of her. It was a very tough one [to write].”

The Woman in Me is due for release on 24 October in the US and UK.
</item 40>

<item 41>
Truth Behind Jake Paul Dating Dillon Danis’ Ex-girlfriend: Did MMA Star Break Up With Savannah Montano? (Essentially Sports, 2023-10-07T21:14:09+00:00)
They haven’t made any claims of breakup and have been silent about their relationship with each other. The only things that have changed are Savannah’s
Instagram feed. She has stopped posting pictures with Danis, which were quite prevalent back in 2020. Also, another thing that hints towards their breakup is
Danis’s first press conference for the Logan Paul fight.

ADVERTISEMENT Article continues below this ad

During the fight, Logan Paul brought up the topic of Montano, when Danis pointedly mentioned their breakup. Paul said, “Yo you’ve been so salty ever since
Savannah rejected your a** to go hook up with real athletes. You don’t gotta take that s**t out on me because my brother hooked up with your ex.” Danis scoffed
at Maverick’s claim and clapped back, “At least I got rid of the sl*t, you’re marrying one.”
</item 41>

<item 42>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Travis’ teammates also showed up to the after-party, as well as his mom, Donna Kelce. The party lasted until 2 am.”

The late-night outing came days after the NFL star officially addressed ongoing dating rumours about him and the “Anti-Hero” singer. During an episode of The
Pat McAfee Show on 21 September, he mocked his brother Jason Kelce for joking that he “100 per cent” believed his younger sibling was dating Swift.

“It’s hilarious how much traction this has actually gotten, right now it’s like a game of telephone,” Kelce said. “Where everybody’s whispering in each other’s
ears and hearing random stuff. No one actually knows what’s going on, especially when you got Jason Kelce on live television telling people just both sides - he
doesn’t know, it’s true, this and that.”

He also noted that he was still open to pursuing a relationship with Swift, revealing that he subtly asked if she’d want to attend one of his football games.
</item 42>

<item 43>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

It may be over a decade since Britney Spears and Kevin Federline ended their marriage, but the former couple continue to have their ups and downs when it comes
to co-parenting their two teenage sons, Sean Preston and Jayden.

Ahead of the release of the pop star’s memoir, The Woman in Me, which is due for release on 24 October, fans are anticipating Spears will share new details
about her relationship with her second husband, Federline.
</item 43>

<item 44>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Britney Spears and Justin Timberlake in 2002. Photograph: Tom Mihalek/AFP/Getty Images

When Timberlake tells the world that she cheated on him, she is all but emblazoned with a scarlet letter. While she knows she is powerless against that
narrative (“I don’t think Justin realised the power he had in shaming me. I don’t think he understands to this day”), him telling the world that they had a
sexual relationship was liberating. “To be honest with you, I liked that Justin said that,” she writes. “Why did my managers work so hard to claim I was some
kind of young-girl virgin even into my 20s? Whose business was it if I’d had sex or not?”

But she still couldn’t escape the cycle that Timberlake set in motion: she describes her 2004 Onyx Hotel tour in the aftermath as “too sexual, for a start.
</item 44>

<item 45>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Spears went on to file for divorce from Federline in November 2006, citing “irreconcilable differences” as the reason for their split. In the couple’s initial
divorce settlement, they were granted joint legal and physical custody over their two children. While it appeared they were on good terms after their
separation, things took a turn in 2018 when Federline requested an increase of $20,000 a month more child support payments from Spears.

After months of court discussions, Federline rejected Spears’s child support proposal and instead doubled his request to $40k a month. A settlement was finally
reached between the two parties in September 2018, with Spears increasing her child support payments to Federline from $20,000 to $35,000 a month.

Federline has since moved on from the relationship, as he married volleyball player Victoria Prince in 2013. They now share two daughters, Jordan, 10, and
Peyton, eight.
</item 45>

<item 46>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
That same day, the fitness instructor officially filed for divorce, listing the date of separation as 28 July 2023 and citing "irreconcilable differences" as
the reason for the split.

In a statement posted online, Asghari said: “After six years of love and commitment to each other my wife and I have decided to end our journey together. We
will hold onto the love and respect we have for each other and I wish her the best always. S*** happens. Asking for privacy seems ridiculous so I will just ask
for everyone including the media to be kind and thoughtful.”

Days before Spears’ memoir release date, Asghari revealed that he had obtained an early copy of the book. In an interview with TMZ on 13 October, he explained
that he wasn’t worried about anything being written in the book that could paint him in a negative light.

“I already read it and I’m very proud of her,” he told the outlet.
</item 46>

<item 47>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Travis Kelce believes the National Football League’s coverage of his rumoured girlfriend Taylor Swift is a bit too much.

Following his game against the New York Jets on 1 October, the 33-year-old Kansas City Chiefs tight end opened up about the NFL broadcast on the most recent
episode of his podcast, New Heights with Jason and Travis Kelce.
</item 47>

<item 48>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“I think some people think that they saw our first date at that game? We would never be psychotic enough to hard launch a first date,” Swift joked.

Back in April, it was revealed that Swift and British actor Joe Alwyn had broken up after six years of dating. The former couple were notoriously private, as
the Conversations with Friends star previously expressed he didn’t want to discuss their relationship because it “feeds into a weird part of the culture”.

Now, Swift has shared that she has nothing to hide in her new relationship with Kelce. “When you say a relationship is public, that means I’m going to see him
do what he loves, we’re showing up for each other, other people are there and we don’t care,” she said. “The opposite of that is you have to go to an extreme
amount of effort to make sure no one knows that you’re seeing someone. And we’re just proud of each other.”
</item 48>

<item 49>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
In the couple’s initial divorce settlement, which was finalised in July 2007, they were granted joint legal and physical custody over their two children.

Singer Britney Spears with ex-husband Kevin Federline in 2006 (Getty Images)

Spears suffered a mental breakdown in 2008 and was placed on a 5150 psychiatric hold, leaving Federline to retain sole custody of their boys. In February of
that year, Spears entered into a conservatorship with her father Jamie Spears, who took complete control over her assets.

Since Spears and Federline’s divorce, it seemed the two were amicably co-parenting their sons when they were seen supporting their boys at their soccer and
football games. Federline moved on to his now wife Victoria Prince. The two share daughters Jordan Kay, 12, and Peyton Marie, nine.

In 2018, things took a turn when Federline requested an increase of $20,000 a month more child support payments from Spears.
</item 49>

<item 50>
Truth Behind Jake Paul Dating Dillon Danis’ Ex-girlfriend: Did MMA Star Break Up With Savannah Montano? (Essentially Sports, 2023-10-07T21:14:09+00:00)
ADVERTISEMENT Article continues below this ad

So, this tells the fans that Danis is no longer together with Montano. Anyway, the fans could look out for her attendance at Danis’s first-ever boxing debut on
October 14 at the AO Arena, Manchester. Although there were rumors about Danis pulling out, it seems the fight is still going to take place. So, her presence at
Danis’s ringside may make things even more complicated. What are your thoughts about this? Are they still together? Tell us in the comments section.

Watch This Story: Dillon Danis’s Relationship with His Wife
</item 50>
=== END ITEM id=b757b0275b139c9f ===

=== ITEM id=750eb038270a4dde ===
Gold sentences and facts:
Sentence 0: Despite the suspension, Jim Harbaugh held his usual weekly news conference Monday, though he mostly avoided addressing the only topic on anyone's mind.
  Fact 0.0: Jim Harbaugh was suspended.
  Fact 0.1: Jim Harbaugh held his usual weekly news conference on Monday despite the suspension.
  Fact 0.2: Jim Harbaugh mostly avoided addressing the only topic on anyone's mind at the news conference.
Sentence 1: It would be Harbaugh's second suspension of the season, as the Wolverines self-imposed a three-game ban on Harbaugh to start the season amid an NCAA investigation that found Harbaugh lied to investigators.
  Fact 1.0: The suspension would be Harbaugh's second suspension of the season.
  Fact 1.1: The Wolverines self-imposed a three-game ban on Harbaugh to start the season.
  Fact 1.2: The self-imposed ban occurred amid an NCAA investigation.
  Fact 1.3: The NCAA investigation found that Harbaugh lied to investigators.
Sentence 2: Harbaugh reportedly watched the game from the team hotel, and now the leading piece of drama returns.
  Fact 2.0: Harbaugh reportedly watched the game from the team hotel.
  Fact 2.1: The leading piece of drama now returns.

Retrieved items:
<item 1>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Michigan players and coaches arrive. No Jim Harbaugh on the bus. pic.twitter.com/wnGzqAxwcp — Adam Rittenberg (@ESPNRittenberg) November 11, 2023

Michigan offensive coordinator Sherrone Moore will serve as acting head coach if Harbaugh is not available.

Friday, Nov. 10

Michigan files temporary restraining order against the Big Ten

In an effort to delay Harbaugh's suspension, the university filed a temporary restraining order in Washtenaw County Trial Court on Friday night. If granted, the
suspension would be delayed, but the clock is ticking before the Wolverines' game vs. Penn State on Saturday.

Michigan's temporary restraining order request has been filed.



Plaintiffs: Jim Harbaugh, U-M regents

Defendants: Big Ten, Tony Petitti



Judicial officer: Timothy P. Connors
</item 1>

<item 2>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Here is Michigan’s 10-page letter to the Big Ten responding to the sportsmanship investigation, reporting with @RossDellenger 1/3 pic.twitter.com/P0oYgIbS67 —
Dan Wetzel (@DanWetzel) November 9, 2023

No timetable for Big Ten decision

The Big Ten is reviewing the "lengthy" official response from Michigan and Jim Harbaugh and it's unclear when a potential ruling might come down, says ESPN's
Pete Thamel. His colleague Adam Rittenberg reported Wednesday that a decision was expected by Thursday.

Source: The Big Ten is reviewing the lengthy written responses from both Michigan and Jim Harbaugh. There’s no indication of any timetable on a decision on
potential Big Ten discipline in the case of Michigan’s alleged illegal signal stealing.
</item 2>

<item 3>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
The Big Ten is also asserting that they are not actually sanctioning Harbaugh, for whom they have no proof of wrongdoing, but that he "embodies the university
for purposes of its football program." pic.twitter.com/S660G6zQaa — Stewart Mandel (@slmandel) November 10, 2023

Jim Harbaugh suspended for remainder of regular season

The Big Ten has officially announced a suspension for Jim Harbaugh for the remaining three games of the regular season, including the Wolverines' rivalry game
with Ohio State.

In a statement, the Big Ten said Michigan conducted an "impermissible, in-person scouting operation over multiple years."

OFFICIAL: Jim Harbaugh suspended for the remaining three regular season games.
</item 3>

<item 4>
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
</item 4>

<item 5>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
sources indicate a Big Ten response is coming early this afternoon. Michigan leaves at 1 and flies midafternoon out of Detroit after getting through security
checks, per usual routine. — Heather Dinich (@CFBHeather) November 10, 2023

Thursday, Nov. 9

Big Ten considering 'range of possibilities' for Harbaugh

ESPN's Adam Rittenberg reported there are a "range of possibilities" the Big Ten is considering as punishment for Harbaugh. The possibility of a suspension has
often garnered most of the headlines, but Rittenberg reported the conference is also considering a significant fine.

While we wait ... although Big Ten has considered a suspension for Michigan coach Jim Harbaugh, it is considering a range of possibilities, including a
significant fine, sources told @CFBHeather and me. Public reprimands/fines are common penalties under B1G sportsmanship policy.
</item 5>

<item 6>
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
</item 6>

<item 7>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Will Harbaugh finally take the parachute to the NFL? Those are questions bound to come up at the end of the 2023 season.

BENDER: Michigan will need Jim Harbaugh against Ohio State

Big Ten

This is the fourth football scandal in less than a calendar year.

Three former Iowa football players and a student manager were charged in an illegal sports wagering scandal. Several student-athletes face allegations of
placing bets on games in which they played in. Northwestern coach Pat Fitzgerald was fired on July 10 after a hazing investigation into the program. Fitzgerald
is suing the university for wrongful termination. Michigan State fired Mel Tucker on Sept. 27 after allegations of sexual misconduct from Brenda Tracy, a
speaker on sexual assault awareness.

Now, Michigan has a scandal on the field, and they countered with allegations that Ohio State and Rutgers leaked Michigan's signs to Purdue in last year's Big
Ten championship game.
</item 7>

<item 8>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
We have to wait for the news, but a suspension is certainly on the table. — Heather Dinich (@CFBHeather) November 10, 2023

If it’s more than 2 games or $10k, the Big Ten will need special permission from an executive board composed of league reps https://t.co/6Z7jb2zqH4 — Heather
Dinich (@CFBHeather) November 10, 2023

Punishment expected on Friday

The college football world will wait with bated breath on Friday as it awaits what Michigan and Jim Harbaugh will face from the Big Ten. ESPN's Heather Dinich
reported the punishment is expected to be announced early Friday afternoon. She noted the team is leaving for State College at 1 p.m. CT for the Penn State game
on Saturday.

sources indicate a Big Ten response is coming early this afternoon.
</item 8>

<item 9>
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
</item 9>

<item 10>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
MORE: Why Jim Harbaugh won't find refuge in the NFL

It was reported earlier in the week that coaches were putting pressure on the Big Ten to hit Michigan with some form of punishment, with Purdue's Ryan Walters
publicly calling for action and claiming the allegations are no longer simply allegations.

Harbaugh was already suspended three games at the start of the season, in a ban imposed by the school itself in an attempt to ward off a separate NCAA
investigation.

Friday, Nov. 3

Suspended football analyst Connor Stalions resigns

Michigan football analyst Connor Stalions, the focus of the NCAA's investigation, resigned on Friday night, the university confirmed.

Statement from Michigan (confirming earlier reports that Connor Stalions is no longer with the program): "Connor Stalions resigned his position with Michigan
Athletics this afternoon. We are unable to comment further regarding this personnel matter."
</item 10>

<item 11>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Will Harbaugh be on the field before the end of the regular season? The Big Ten made a Friday ruling that will prohibit Harbaugh from being on the field for the
final three regular-season games as a result of the in-person scouting and sign-stealing scandal that led to the resignation of staffer Conor Stallions. ESPN
reported there will be a temporary restraining order hearing on Nov. 17 that will answer that question.

On the field, Michigan answered the question about its College Football Playoff. Sort of. The Wolverines proved they could beat Penn State (8-2, 5-2) in a
20-yard box. Neither team completed a pass of 20 yards or more. Michigan trailed 3-0 after the first quarter. The Wolverines had eight carries for 10 yards, and
Penn State's Abdul Carter set the tone with a jarring hit on Blake Corum that nearly resulted in a fumble.

Michigan took control with the running game.
</item 11>

<item 12>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Within it, the school warned Big Ten commissioner Tony Pettiti against levying "premature" disciplinary action against Jim Harbaugh and alleged that in-person
scouting and sign-stealing are commonplace across the NCAA.

"The conference should act cautiously when setting precedent given the reality that in-person scouting, collusion among opponents, and other questionable
practices may well be far more prevalent than believed," the letter read.

Michigan also included with the letter "two attached documents of Michigan's offensive and defensive signs allegedly stolen and shared by other Big Ten teams,"
per Yahoo Sports.

Conference awaits Michigan response

The Big Ten expects a formal response from Michigan to the notice of discipline by the end of the day Wednesday, ESPN's Pete Thamel reports. His colleague Adam
Rittenberg adds that a ruling is expected to come down Thursday.

Sources: The Big Ten expects a response from Michigan on the notice of discipline by the end of day on Wednesday. Don’t expect any ruling by the Big Ten on
Wednesday, as they are expected to take time to absorb Michigan’s response.
</item 12>

<item 13>
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
</item 13>

<item 14>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
12

U-M president praises Wolverines for handling 'challenges and adversity'

The day after the Penn State win, University of Michigan President Santa Ono posted on social media, praising the team's ability to "respond to any challenge
head on with a conviction to do better and to emerge even stronger."

Countless members of the University of Michigan family have reached out to me over the weekend and I wanted to express my appreciation. Like any community, we
face our share of challenges and adversity. There have been many such moments in our history. But as our team showed so… — Santa Ono (@SantaJOno) November 12,
2023

The post drew immediate and extensive blowback on social media and beyond, as observers questioned Ono's tone with Michigan in the midst of a self-inflicted
controversy.

Saturday, Nov. 11

Warde Manuel issues strong statement against Big Ten

Michigan athletic director Warde Manuel did not mince words in a statement on the Big Ten's suspension of Harbaugh.
</item 14>

<item 15>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
🗣️ "Bet" pic.twitter.com/ENB7O8MyPv — The Sporting News (@sportingnews) November 10, 2023

Michigan announces intention to dispute suspension

Michigan has released a statement expressing frustration with the Big Ten's process and says it plans to seek a court order to block Harbaugh's suspension from
taking effect.

NEWS: Michigan says it intends to seek a court order with Jim Harbaugh preventing disciplinary action from taking effect.



Full statement: pic.twitter.com/p92yUCLsTX — Matt Fortuna (@Matt_Fortuna) November 10, 2023

Big Ten claims no evidence Harbaugh was aware of scheme

While the Big Ten says it's "proven" that an illegal sign-stealing scheme existed, the conference could not prove that Harbaugh knew about it. So, why was
Harbaugh suspended? The Big Ten asserts that, because he is the head coach, he has a responsibility for preventing this kind of scheme from taking place.
</item 15>

<item 16>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
However, the primary difference is Michigan is accused of using in-person scouting, which has been prohibited under NCAA rules for nearly 30 years, to acquire
its information.

Monday, Nov. 6

No connection between Buckeyes coach, Michigan investigation

Ohio State coach Ryan Day and his family have no connection to the Big Ten and NCAA investigations into Michigan, Yahoo Sports reported.

A report emerged over the weekend indicating Day may have had some involvement in the probe into the Buckeyes' arch-rivals, but the NCAA told the Big Ten there
are no ties that it knows of between the Ohio State coach and the investigation.

NCAA shares findings with Big Ten as Harbaugh suspension looms

Yahoo Sports reported Monday that the NCAA shared its investigation findings with the Big Ten, noting that the findings "do not connect the in-person scouting
and recording of opponents’ sidelines to Harbaugh."

Still, the conference is considering a suspension for Harbaugh and could hand down punishments before the end of the week.
</item 16>

<item 17>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
On the other, he could emerge as a martyr for Michigan fans given there is reportedly no proof yet he knew of Stalion's rule-breaking.

Harbaugh will have a temporary restraining order hearing on Nov. 17, according to ESPN.com. That will determine whether he returns to the field for the final
two games against Maryland and No. 1 Ohio State.

It's an inescapable fact that Harbaugh is 31-3 – and Michigan is 35-3 – since the program bottomed out at the end of the 2020 season. While it is almost
impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons, it will be a tough
stain to erase knowing those violations are now on the Big Ten record.

What will Harbaugh do now? That is almost impossible to predict given his combative personality. Would he sign a contract extension and continue to fight the
NCAA? Will Harbaugh finally take the parachute to the NFL?
</item 17>

<item 18>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Did Michigan really need Jim Harbaugh on Saturday?

Perhaps the result provides the short-term answer. No. 3 Michigan beat No. 10 Penn State 24-15 at Beaver Stadium on Saturday. The Wolverines threw one pass in
the second half – and that didn't even technically count – in a risk-averse game plan under acting coach Sherrone Moore. The defense smothered the Nittany
Lions, and that will lead to more buildup ahead of the showdown with No. 1 Ohio State on Nov. 25.

Then Moore let it all there in the Fox postgame interview with Jenny Taft. Even if the answer is no, Harbaugh was ever-present.

"I want to thank the Lord and I want to thank Coach Harbaugh," Moore said. "I f------ love you man. I love the s--- out of you, man. I did this for you, for
this university, the president, our AD.
</item 18>

<item 19>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Judicial officer: Timothy P. Connors



Connors is a U-M lecturer, adjunct professor at Wayne State and Vermont. pic.twitter.com/WR4G4WTtfH — Chris Vannini (@ChrisVannini) November 11, 2023

Michigan players respond to Harbaugh suspension

J.J. McCarthy and a handful of other Michigan players tweeted "Bet" at the same time on Friday night, hours after Harbaugh was suspended for the remainder of
the regular season. McCarthy wore a shirt that said "Free Harbaugh" when the 59-year-old was serving a separate suspension at the start of the season.

Michigan players had a simple message on social media following Jim Harbaugh's suspension 👀
</item 19>

<item 20>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
He initially told reporters he would speak during a scheduled hearing Friday, but then backtracked a bit and sounded as if he wasn't committed to that course,
pending legal advice.

Harbaugh said he will speak at the hearing on Friday. He said he’s looking for “due process.” Mentioned a civics class he took 40 years ago. “You’re innocent
until proven guilty. I’d like that opportunity.” — Nicole Auerbach (@NicoleAuerbach) November 13, 2023

Harbaugh clarifies that he’s not sure whether he’ll testify Friday. “That’s not my dance floor,” he says. “I’ve watched a lot of shows. I’ve watched Judge Judy
a lot.” — Austin Meek (@byAustinMeek) November 13, 2023

Sunday, Nov.
</item 20>

<item 21>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
“We went through and made sure we asked all the questions and got our compliance people involved. None of that came back at all. I can answer very strongly that
that did not happen.”

Michigan submits response to Big Ten

In the latest step toward a resolution, Michigan has submitted its response to the Big Ten's notice of discipline. ESPN reported earlier Wednesday that no
ruling from the conference is expected until Thursday, as the Big Ten is planning to "take time to absorb" what Michigan has to say.

Michigan has sent its response to the Big Ten, a school official tells ESPN. Commissioner Tony Petitti now must decide how to proceed with potential discipline.
— Adam Rittenberg (@ESPNRittenberg) November 8, 2023

Yahoo Sports obtained a copy of the 10-page letter sent by Michigan to the Big Ten.
</item 21>

<item 22>
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
</item 22>

<item 23>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Who is the good guy? Who is the villain? Ohio State and Michigan locked in their answers a long time ago. More than 17 million people watched last year's
matchup – a 45-23 Michigan victory where Stalions' involvement continues to be scrutinized through a double-barrel Big Ten and NCAA investigation. Big Ten
commissioner Tony Petitti's ruling sent a shockwave through the conference Friday, mainly because the third game means Harbaugh will not be on the sideline for
The Game. Early forecasts say 20 million will tune in for The Game this year.

Will Harbaugh really not be there? That is for the courts to decide, but Moore’s comments suggest Harbaugh, president Santa Ono and athletic director Warde
Manuel are on a united front; consequences be damned in 2024 and beyond.

In the short term? The College Football Playoff committee could slide Michigan up to No. 2 now, and the debate with the Buckeyes and No.
</item 23>

<item 24>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Pete Thamel (@PeteThamel) November 8, 2023

Tuesday, Nov. 7

Michigan says three other Big Ten schools shared decoded Wolverines signals

While Michigan tries to navigate the storm around its program, the Wolverines are also trying to make one thing clear: they're hardly alone in trying to
decipher opponents' signals.

Michigan has shared evidence with the Big Ten and NCAA showing staff members at Ohio State, Purdue and Rutgers shared intelligence on the Wolverines' signals
prior to last year's Michigan-Purdue Big Ten Championship Game, The Athletic reported.

Documents breaking down Michigan's signals for numerous plays were compiled by Purdue staffers ahead of the game based on information provided by Ohio State and
Rutgers football personnel, the report says. The person who supplied the documents told The Athletic he spent 10 to 12 hours per week trying to decode signals
based on game video and said the practice of sharing decoded signals with other schools is common.
</item 24>

<item 25>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
pic.twitter.com/Qm9iehYVTR — Pete Thamel (@PeteThamel) November 11, 2023

Harbaugh will not be on the sidelines vs. Penn State

Michigan filed a temporary restraining order against the Big Ten on Friday night, hoping to block Harbaugh's suspension. But there will not be a ruling before
the Wolverines' game vs. Penn State, with the hearing scheduled for Friday, Nov. 17.

No ruling on Michigan’s TRO filing today, so Jim Harbaugh will not coach vs. Penn State.



Statement from the school: “We look forward to presenting our case next week where we intend to demonstrate that the Big Ten has not acted legally or fairly.”
</item 25>

<item 26>
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
</item 26>

<item 27>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Jim Harbaugh’s attorney, Tom Mars, tells ESPN he fully expects Jim Harbaugh to fly with the team to Penn State. “I’m confident he’s going to be on that plane
one way or another," Mars said. — Pete Thamel (@PeteThamel) November 10, 2023

Punishment options for Michigan

Dinich has reported the punishment could be anything, ranging from "indefinite suspension to public reprimand or lofty fine." However, she also reported that
there is potentially a limit to what the conference can do without permission from an executive board. Anything more than two games or a fine of $10,000 would
require special permission.

Please remember the Big Ten has considered a WIDE range of punishments - everything from indefinite suspension to public reprimand or lofty fine, sources tell
me n @ESPNRittenberg … no assumptions. We have to wait for the news, but a suspension is certainly on the table.
</item 27>

<item 28>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Yahoo says a multi-game suspension for Harbaugh is on the table, though Michigan is fully expected to fight a ban if one is handed down. It would be Harbaugh's
second suspension of the season, as the Wolverines self-imposed a three-game ban on Harbaugh to start the season amid an NCAA investigation that found Harbaugh
lied to investigators.

Saturday, Nov. 4

Harbaugh 'likely' to be suspended

According to The Athletic's Pete Thamel, the Big Ten might be getting closer to handing down a suspension for Harbaugh.

"Over the past few days, Petitti has spoken with the athletic directors, he's spoken with the presidents, he's spoken with the coaches. And the next steps
appear that there will be some type of punishment for Michigan. The most likely scenario here is some type of suspension for Jim Harbaugh," Thamel said on
ESPN's "College GameDay."
</item 28>

<item 29>
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
</item 29>

<item 30>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Public reprimands/fines are common penalties under B1G sportsmanship policy. — Adam Rittenberg (@ESPNRittenberg) November 9, 2023

Michigan's 10-page letter to Big Ten released

Yahoo Sports' Dan Wetzel posted Michigan's full 10-page letter to the Big Ten on X (formerly Twitter). The letter outlines Michigan's belief that the Big Ten
would be acting too hastily in suspending Harbaugh or levying any punishment toward Michigan before the completion of a full NCAA investigation.

In the letter, Michigan said the university has not had the chance to review all the evidence and respond, which oversteps the due process allowed for Michigan.
It said Pettiti should not have unilateral authority to punish violations without first having a completed investigation and then a response from the Compliance
and Reinstatement Committee, which would recommend penalties for rules violations.
</item 30>
=== END ITEM id=750eb038270a4dde ===
