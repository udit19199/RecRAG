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

=== ITEM id=f3fc7b19aa5c631a ===
Gold sentences and facts:
Sentence 0: The guidance lays out ways to responsibly use AI in the courts, including being aware of potential bias and upholding privacy.
  Fact 0.0: The guidance lays out ways to use AI responsibly in the courts.
  Fact 0.1: The guidance says courts should be aware of potential bias when using AI.
  Fact 0.2: The guidance says courts should uphold privacy when using AI.
Sentence 1: They promised “invest[ment] in the AI-powered technology” that, amongst other things, would help YouTube “protect our community of viewers, creators, artists and songwriters… [from] trademark and copyright abuse” .
  Fact 1.0: The party referred to as 'they' promised investment in the AI-powered technology.
  Fact 1.1: The AI-powered technology would, among other things, help YouTube protect its community of viewers, creators, artists and songwriters.
  Fact 1.2: The AI-powered technology would help YouTube protect its community from trademark and copyright abuse.

Retrieved items:
<item 1>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
But while a few egregious examples of misuse will be punished, a lack of relevant laws specific to it means that it will necessarily only haphazardly be brought
to court. On that note…

Early adopters take new rules by the horns

Big moves like the EU’s AI Act could change how the industry works, but they tend to be slow to take effect. That’s by design, so companies don’t have to adjust
to new rules overnight, but it also means that we won’t see the effect of these big laws for a good while except among those willing to make changes
preemptively and voluntarily. There will be a lot of “we are beginning the process of…” talk. (Also expect a few quiet lawsuits challenging various parts of
laws.)

To that end we can expect a newly flourishing AI compliance industry as the billions going into the technology prompt matching investments (at a smaller scale,
but still considerable) in making sure the tools and processes meet international and local standards.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, AI, 8 predictions for AI in 2024, EU’s AI Act, AI compliance industry, international and local standards
</item 1>

<item 2>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT can now browse the internet to provide you with current and authoritative information, complete with direct links to sources. It is no longer limited to
data before September 2021. pic.twitter.com/pyj8a9HWkB — OpenAI (@OpenAI) September 27, 2023

ChatGPT now has a voice

OpenAI announced that it’s adding a new voice for verbal conversations and image-based smarts to the AI-powered chatbot.

Poland opens an investigation against OpenAI

The Polish authority publically announced it has opened an investigation regarding ChatGPT — accusing the company of a string of breaches of the EU’s General
Data Protection Regulation (GDPR).

OpenAI unveils DALL-E 3

The upgraded text-to-image tool, DALL-E 3, uses ChatGPT to help fill in prompts.

Graph facts:
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> EU
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> EU
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> EU
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> EU
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> EU

Entities: EU, ChatGPT, Polish authority, investigation regarding ChatGPT, EU’s General Data Protection Regulation (GDPR), ChatGPT: Everything you need to know
about the AI-powered chatbot, TechCrunch, OpenAI, DALL-E 3, Poland
</item 2>

<item 3>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Legal cases build and break

We saw a fair number of lawsuits filed in 2023, but few saw any real movement, let alone success. Most suits over copyright and other missteps in the AI
industry are still pending. 2024 will see a lot of them fall by the wayside, as companies stonewall critical information like training data and methods, making
allegations like the use of thousands of copyrighted books difficult to prove in court.

This was only the beginning, however, and many of these lawsuits were filed essentially on principle. Though they may not succeed, they may crack the process
open far enough during testimony and discovery that companies would rather settle than have certain information come to light. 2024 will bring new lawsuits as
well, ones pertaining to misuse and abuse of AI, such as wrongful termination, bias in hiring and lending, and other areas where AI is being put to work without
a lot of thought.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> copyright
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> wrongful termination
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> training data
- Pat Fitzgerald -ACCUSED_OF-> wrongful termination
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> copyright

Entities: training data, wrongful termination, copyright, TechCrunch, AI, 8 predictions for AI in 2024, misuse and abuse of AI, bias in hiring, bias in lending
</item 3>

<item 4>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
OpenAI promises to defend business customers against copyright claims

Bowing to peer pressure, OpenAI it will pay legal costs incurred by customers who face lawsuits over IP claims against work generated by an OpenAI tool. The
protections seemingly don’t extend to all OpenAI products, like the free and Plus tiers of ChatGPT.

As OpenAI’s multimodal API launches broadly, research shows it’s still flawed

OpenAI announced that GPT-4 with vision will become available alongside the upcoming launch of GPT-4 Turbo API. But some researchers found that the model
remains flawed in several significant and problematic ways.

OpenAI launches API, letting developers build ‘assistants’ into their apps

At its OpenAI DevDay, OpenAI announced the Assistants API to help developers build “agent-like experiences” within their apps. Use cases range from a natural
language-based data analysis app to a coding assistant or even an AI-powered vacation planner.
</item 4>

<item 5>
Graph query row:
article: Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office.
published_on: 2023-11-23T18:48:48+00:00
connection: PUBLISHED_BY
related_entity: Music Business Worldwide
stance_evidence: Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright
Office. (Music Business Worldwide, 2023-11-23T18:48:48+00:00)
What’s happened?

MBW Explains is a series of analytical features in which we explore the context behind major music industry talking points – and suggest what might happen next.
MBW Explains is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

As we said last week: Welcome to the future.

Seven days ago (November 16), YouTube unveiled a revolutionary new experiment – ‘Dream Track’ – enabling creators to clone the vocals, via AI tech and with
official consent, of well-known stars.

Charlie Puth’s voice is on there. John Legend’s voice is on there. Demi Lovato’s voice is on there.

In fact, the voices of a number of well-known recording artists signed to Warner Music Group and Universal Music Group – or labels within each major – are on
there.
</item 5>

<item 6>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
“ChatGPT Business will follow our API’s data usage policies, which means that end users’ data won’t be used to train our models by default,” OpenAI wrote in a
blog post. “We plan to make ChatGPT Business available in the coming months.”

OpenAI wants to trademark “GPT”

OpenAI applied for a trademark for “GPT,” which stands for “Generative Pre-trained Transformer,” last December. Last month, the company petitioned the USPTO to
speed up the process, citing the “myriad infringements and counterfeit apps” beginning to spring into existence.

Unfortunately for OpenAI, its petition was dismissed last week. According to the agency, OpenAI’s attorneys neglected to pay an associated fee as well as
provide “appropriate documentary evidence supporting the justification of special action.”

That means a decision could take up to five more months.

Graph facts:
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> GPT
- The people who ruined the internet -MENTIONS-> GPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT

Entities: GPT, ChatGPT, ChatGPT: Everything you need to know about the AI-powered chatbot, ChatGPT Business, USPTO, GPT trademark application, Petition to speed
up the trademark process, Dismissal of OpenAI's petition, API, TechCrunch, AI, OpenAI
</item 6>

<item 7>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Italy’s data protection authority has just put out a timely reminder that some countries do have laws that already apply to cutting edge AI: it has ordered
OpenAI to stop processing people’s data locally with immediate effect. The Italian DPA said it’s concerned that the ChatGPT maker is breaching the European
Union’s General Data Protection Regulation (GDPR), and is opening an investigation.

March 2023

1,100+ signatories signed an open letter asking all ‘AI labs to immediately pause for 6 months’

The letter’s signatories include Elon Musk, Steve Wozniak and Tristan Harris of the Center for Humane Technology, among others. The letter calls on “all AI labs
to immediately pause for at least 6 months the training of AI systems more powerful than GPT-4.”

The letter reads:

Contemporary AI systems are now becoming human-competitive at general tasks,[3] and we must ask ourselves: Should we let machines flood our information channels
with propaganda and untruth?

Graph facts:
- One year later, ChatGPT is still alive and kicking -MENTIONS-> Steve Wozniak
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> Steve Wozniak
- Steve Wozniak -SAID-> Six month moratorium on AI development
- Steve Wozniak -PARTICIPATED_IN-> Signing an open letter calling for the immediate pause of large-scale AI experiments
- Steve Wozniak -PARTICIPATED_IN-> Signing an open letter demanding a six month moratorium on AI development

Entities: Steve Wozniak, ChatGPT, ChatGPT: Everything you need to know about the AI-powered chatbot, Italian DPA, General Data Protection Regulation, Open
letter calling for a pause in AI training, Tristan Harris, Center for Humane Technology, AI labs, Investigation into OpenAI, AI systems, TechCrunch, GPT-4, Elon
Musk, European Union
</item 7>

<item 8>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
On iPad, ChatGPT now runs in full-screen mode, optimized for the tablet’s interface.

May 2023

Texas judge orders all AI-generated content must be declared and checked

The Texas federal judge has added a requirement that any attorney appearing in his court must attest that “no portion of the filing was drafted by generative
artificial intelligence,” or if it was, that it was checked “by a human being.”

ChatGPT app expanded to more than 30 countries

The list of new countries includes Algeria, Argentina, Azerbaijan, Bolivia, Brazil, Canada, Chile, Costa Rica, Ecuador, Estonia, Ghana, India, Iraq, Israel,
Japan, Jordan, Kazakhstan, Kuwait, Lebanon, Lithuania, Mauritania, Mauritius, Mexico, Morocco, Namibia, Nauru, Oman, Pakistan, Peru, Poland, Qatar, Slovenia,
Tunisia and the United Arab Emirates.

Graph facts:
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Argentina
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Argentina
- Watch Argentina vs Paraguay live stream, TV channel, lineups, betting odds for World Cup qualifying match -MENTIONS-> Argentina
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> Argentina
- Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list -MENTIONS-> Argentina

Entities: Argentina, Brazil, ChatGPT, ChatGPT: Everything you need to know about the AI-powered chatbot, Texas federal judge, Texas judge orders all
AI-generated content must be declared and checked, Algeria, Azerbaijan, Bolivia, Ecuador, Estonia, Kazakhstan, Lithuania, Mauritania, Mauritius
</item 8>

<item 9>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
In its privacy policy, the ChatGPT maker makes a passing acknowledgement of the objection requirements attached to relying on “legitimate interest” (LI),
pointing users towards more information about requesting an opt out — when it writes: “See here for instructions on how you can opt out of our use of your
information to train our models.”

What controversies have surrounded ChatGPT?

Recently, Discord announced that it had integrated OpenAI’s technology into its bot named Clyde where two users tricked Clyde into providing them with
instructions for making the illegal drug methamphetamine (meth) and the incendiary mixture napalm.

An Australian mayor has publicly announced he may sue OpenAI for defamation due to ChatGPT’s false claims that he had served time in prison for bribery. This
would be the first defamation lawsuit against the text-generating service.

CNET found itself in the midst of controversy after Futurism reported the publication was publishing articles under a mysterious byline completely generated by
AI.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, ChatGPT: Everything you need to know about the AI-powered chatbot, Clyde, methamphetamine, napalm, Australian mayor, CNET, Futurism, bribery,
TechCrunch, defamation, AI, OpenAI, Discord
</item 9>

<item 10>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
are now allowed to use ChatGPT in legal rulings

The U.K. Judicial Office issued guidance that permits judges to use ChatGPT, along with other AI tools, to write legal rulings and perform court duties. The
guidance lays out ways to responsibly use AI in the courts, including being aware of potential bias and upholding privacy.

OpenAI makes repeating words “forever” a violation of its terms of service after Google DeepMind test

Following an experiment by Google DeepMind researchers that led ChatGPT to repeat portions of its training data, OpenAI has flagged asking ChatGPT to repeat
specific words “forever” as a violation of its terms of service.

Lawmakers in Brazil enact an ordinance written by ChatGPT

City lawmakers in Brazil enacted a piece of legislation written entirely by ChatGPT without even knowing. Weeks after the bill was passed, Porto Alegre
councilman Ramiro Rosário admitted that he used ChatGPT to write the proposal, and did not tell fellow council members until after the fact.
</item 10>
=== END ITEM id=f3fc7b19aa5c631a ===

=== ITEM id=c9ab6b58bc1306e4 ===
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

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Sundar Pichai
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sundar Pichai
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Sundar Pichai
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Sundar Pichai
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Sundar Pichai

Entities: Sundar Pichai, Department of Justice, antitrust, Apple, online search, TechCrunch, New York, California, U.S. government, digital advertising, U.S.
government’s antitrust trial against Google, 2020 antitrust claim, January filing, eight states, 5 things we learned so far about the Google antitrust case
</item 4>

<item 5>
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
</item 5>

<item 6>
Graph query row:
article: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line
published_on: 2023-12-15T17:56:02+00:00
publishers: ['TechCrunch']
central_claims: ['anticompetitive behavior', 'U.S. antitrust law', 'Sherman Act', 'question-and-answer technologies', 'misappropriation of Publishers’ content',
'app store antitrust issues', 'digital ad technologies', 'Clayton Act', 'search and search advertising', 'digital news and reference publishing', 'class-wide
monetary and injunctive relief', 'free marketplace of ideas', 'free press', 'antitrust', 'consumers', 'AI', 'AI technologies', 'artificial intelligence',
'labor']
mentioned_organizations: ['Alphabet', 'Google', 'OpenAI']
allegations: ['anticompetitive behavior', 'U.S. antitrust law', 'Sherman Act', 'US v Google trial', 'class action antitrust suit', 'misappropriation of
Publishers’ content', 'Epic Games lawsuit against Google over app store antitrust issues', 'class action antitrust suit against Google', 'digital ad
technologies', 'anti-competitive practices', 'search monopoly', 'antitrust', 'monopoly', 'anti-competition law', 'systematically destroying chats',
'anticompetitive conduct', 'antitrust law', 'unlawful monopoly', 'making potential evidence disappear', 'illegal tying', 'Android app distribution', 'Android
in-app payment processing', 'deleting potentially incriminating messages', 'EU’s General Data Protection Regulation (GDPR)', 'General Data Protection
Regulation', 'consumer protection laws', 'copyright']
connected_articles: ['report on Thursday']
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 7>

<item 8>
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

Graph facts:
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Chrome
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Chrome
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> Chrome
- ‘Mass breach of privacy’: TikTok under fire for tracking users online -MENTIONS-> Chrome
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Chrome

Entities: Chrome, Mozilla, Apple, TechCrunch, Prabhakar Raghavan, Pichai, trial hearing, court testimony, Google antitrust case, decision made with the consumer
in mind, 5 things we learned so far about the Google antitrust case, Samsung, Google, New York Times, iPhones
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=c9ab6b58bc1306e4 ===

=== ITEM id=9269343ee1e09b9e ===
Gold sentences and facts:
Sentence 0: You can find value in player and team props by gathering information from multiple sources and tracking how players and their teammates perform.
  Fact 0.0: Value can be found in player and team props.
  Fact 0.1: Value in player and team props can be found by gathering information from multiple sources.
  Fact 0.2: Value in player and team props can be found by tracking how players and their teammates perform.
Sentence 1: These can be team or individual player prop bets, such as which team is winning at halftime or the total number of receptions a player collects.
  Fact 1.0: Props can be team prop bets.
  Fact 1.1: Props can be individual player prop bets.
  Fact 1.2: A prop bet can be on which team is winning at halftime.
  Fact 1.3: A prop bet can be on the total number of receptions a player collects.

Retrieved items:
<item 1>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Player props could focus on how many points a player scores or how many rebounds he pulls down. These are typically provided in an over/under format.

Then, team props can range from which team will score the first bucket to the total points scored in a quarter to which team will lead at halftime. The
selection is often robust, especially when high-profile teams play.

Futures bets are exactly what their name implies: bets on events that will happen in the future. Common futures markets include:

and much more. You can usually bet on NBA futures odds throughout the year, including well before the season starts and throughout the regular season. The
further away from the event, the more likely you will find heavier odds in your favor.

Is NBA betting legal in the U.S.?

In the US, the landscape of sports betting, specifically NBA betting, has evolved considerably. While NBA betting is permitted, its legality varies across the
nation. The significant turning point came in 2018 when the US Supreme Court made a pivotal decision.

Graph facts:
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> United States
- The best movies of 2023 so far -MENTIONS-> United States
- Mattel slammed for ‘inaccurate’ Cherokee Barbie -MENTIONS-> United States
- Watch Argentina vs Paraguay live stream, TV channel, lineups, betting odds for World Cup qualifying match -MENTIONS-> United States
- Israel's defense undone by reliance on technology -MENTIONS-> United States

Entities: United States, Sporting News, NBA betting, The Best NBA Betting Sites and Apps for the 2023-24 Season, US Supreme Court, 2023-24 Season, 2018 pivotal
decision, futures bets, NBA, player props, team props, sports betting
</item 1>

<item 2>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
You’ll often come across NBA team prop bets that are based on a team’s points, such as the first team to reach 20 points, the first team to score in a game, or
whether the total points by both teams in a contest will end up being odd or even.

Team props, also known as “game props,” don’t rely on individual players but rather on the team as a whole. Player props, on the other hand, are driven by
individual performance.

These individual props are focused on statistical milestones during a game. The most popular bets typically focus on points, rebounds, assists, and three-point
field goals. You can also bet on combination wagers, such as points and rebounds together.

In most cases, you can bet on traditional statistics for the five starters on each team.

See more:

Tips for betting on NBA props

Every bettor should develop their strategies, but we do have three tips you can use for NBA props.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA, player props, NBA Prop Bets - Best Team and Player Props 2023/24 Season, team prop bets, statistical milestones, points, rebounds,
assists, three-point field goals, combination wagers, traditional statistics, NBA props
</item 2>

<item 3>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
The NBA is one of the most popular sports to bet on in the US. NBA prop bets are an easy way to jump into basketball betting.

Proposition bets, or props, are easy to understand, numerous, and cover both player props and team props to give you plenty of variety.

Below, we look at betting NBA props at the best NBA betting sites and share some key information to make informed bets.

What are NBA props?

A prop bet is a side wager on an aspect of a game that usually doesn’t tie directly to the result. In the NBA, prop bets deal with both players and teams.

Graph facts:
- ASX set for bright start despite Wall Street slip; $A drops -MENTIONS-> US
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> US
- Is Google Search better than the rest? And is that fair? -MENTIONS-> US
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> US
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> US

Entities: US, Sporting News, NBA, basketball betting, proposition bets, player props, team props, NBA betting sites, NBA Prop Bets - Best Team and Player Props
2023/24 Season
</item 3>

<item 4>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
In the NBA, prop bets deal with both players and teams.

Some examples of NBA player props are:

How many points a player will score

Number of rebounds a player will have

How many assists a player will get

Rebounds and assists combined

Points, rebounds, and assists combined

How many steals a player will get

Number of three-pointers made

Some examples of NBA team props include:

Which team will reach 10 points first

Which team will have the highest-scoring period

Which team will score first

Which team will win the first half

Which team will win the second half

Will there be a fourth-quarter comeback victory

All of the best sports betting sites have their selection of prop bets for each NBA game, so shop around to find the sportsbooks that best suit your needs.

How to bet on NBA props

The first step to betting on NBA props is opening an account at an online sportsbook. The process is simple, free, and doesn’t take very long.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA, prop bets, sports betting, sportsbook, NBA game, NBA Prop Bets - Best Team and Player Props 2023/24 Season
</item 4>

<item 5>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Pay attention to player and team news: Oddsmakers use various information to develop their lines for games and players, and you should, too. Keep tabs on player
news and injuries, when top players are being rested, how they performed in recent games, and more. You can find value in player and team props by gathering
information from multiple sources and tracking how players and their teammates perform. For example, if one of the starting guards is questionable for a game,
you might want to consider the other guard for increased points or assists since they’ll likely handle a heavier load.

Keep an eye on usage and minutes: Even slight increases to a player’s average minutes can make a difference when betting player props, especially later in the
season when aging veteran players tend to rest more. Keep yourself informed on usage and minutes trends for the players you’re considering betting on, and use
them to your advantage.

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA, player props, team props, NBA Prop Bets - Best Team and Player Props 2023/24 Season, usage and minutes, betting
</item 5>

<item 6>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Remember that some of the best players in the NBA were once on the bench for most games until small increases in their playing time gave them a chance to shine.

Shop for the best lines: This tip applies to every bet you make. Using multiple sportsbook accounts lets you find the best odds for every NBA prop bet. Even the
smallest differences, like one sportsbook having odds at +125 and another at +135, can make a difference in your profit. By shopping for the best lines, you can
make smarter bets and find the best value.

Related Pages:

Graph facts:
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -PUBLISHED_BY-> Sporting News
- North Texas vs. SMU odds, props, predictions: Red hot Mustangs could overpower not-so-Mean Green -PUBLISHED_BY-> Sporting News
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -PUBLISHED_BY-> Sporting News
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -PUBLISHED_BY-> Sporting News
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -PUBLISHED_BY-> Sporting News

Entities: Sporting News, NBA, NBA prop bets, NBA Prop Bets - Best Team and Player Props 2023/24 Season
</item 6>

<item 7>
Graph query row:
article: NBA Prop Bets - Best Team and Player Props 2023/24 Season
published_on: 2023-10-02T23:49:35+00:00
publisher: Sporting News
connected_details: ['2023/24 season', 'US', 'NBA', 'basketball betting', 'proposition bets', 'player props', 'team props', 'NBA betting sites', 'prop bets',
'sports betting', 'sportsbook', 'NBA game', 'NBA prop bets', 'Team and player props', 'online sports betting', 'online sportsbook', 'terms and conditions',
'welcome bonus', 'deposit', 'e-wallet', 'basketball', 'National Basketball Association', 'team prop bets', 'statistical milestones', 'points', 'rebounds',
'assists', 'three-point field goals', 'combination wagers', 'traditional statistics', 'NBA props', 'usage and minutes', 'betting', 'Giannis Antetokounmpo',
'LeBron James']
passages: ['NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nThe NBA is one of the most popular sports to
bet on in the US. NBA prop bets are an easy way to jump into basketball betting.\n\nProposition bets, or props, are easy to understand, numerous, and cover both
player props and team props to give you plenty of variety.\n\nBelow, we look at betting NBA props at the best NBA betting sites and share some key information
to make informed bets.\n\nWhat are NBA props?\n\nA prop bet is a side wager on an aspect of a game that usually doesn’t tie directly to the result. In the NBA,
prop bets deal with both players and teams.', 'NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nIn the NBA,
prop bets deal with both players and teams.\n\nSome examples of NBA player props are:\n\nHow many points a player will score\n\nNumber of rebounds a player will
have\n\nHow many assists a player will get\n\nRebounds and assists combined\n\nPoints, rebounds, and assists combined\n\nHow many steals a player will
get\n\nNumber of three-pointers made\n\nSome examples of NBA team props include:\n\nWhich team will reach 10 points first\n\nWhich team will have the
highest-scoring period\n\nWhich team will score first\n\nWhich team will win the first half\n\nWhich team will win the second half\n\nWill there be a
fourth-quarter comeback victory\n\nAll of the best sports betting sites have their selection of prop bets for each NBA game, so shop around to find the
sportsbooks that best suit your needs.\n\nHow to bet on NBA props\n\nThe first step to betting on NBA props is opening an account at an online sportsbook. The
process is simple, free, and doesn’t take very long.', 'NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News,
2023-10-02T23:49:35+00:00)\nThe process is simple, free, and doesn’t take very long. Below are the steps to opening a new account at an online
sportsbook.\n\nChoose your sportsbook: Any state with legal online sports betting has a variety of sportsbooks. If you aren’t sure which sportsbooks best fit
your needs, research and read our reviews on each. And don’t forget that you can have an account with more than one. Once you decide which sportsbook you’re
interested in, follow the link from our site to qualify for the best new customer welcome bonus.\n\nRegister for an account: At the top of the homepage of the
sportsbook website, you will find a button that says “register” or “sign up.” Click that, and you’ll be taken to a form you must fill out. It will ask for
personal information such as your name, phone number, email address, last four digits of your Social Security number, and more.', 'NBA Prop Bets - Best Team and
Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nAccept the terms and conditions: Before finalizing your account information, you must
accept the sportsbook’s terms and conditions. We recommend you read through these to understand the rules your sportsbook will enforce regarding bets,
especially for the sports you intend to wager on. The terms and conditions are also available on the sportsbook website for you to read at any point.\n\nGet
your bonus: At this point, you can accept your welcome bonus. Be sure to read through the bonus requirements before accepting anything. If a bonus code is
needed, which you can find on our website, this is where you will input it during the registration process.\n\nMake a deposit: Now that you’ve signed up for an
account, you can start funding your bankroll by making a deposit. Most legal sportsbooks have a variety of ways you can add funds to your account, such as
through your bank or an e-wallet.\n\nStart betting: Go to the basketball section of your sportsbook and select the NBA.', 'NBA Prop Bets - Best Team and Player
Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nStart betting: Go to the basketball section of your sportsbook and select the NBA. From there,
you can choose individual games and make bets like money lines, point spreads, totals, props, futures, and more.\n\nHow to read NBA props odds\n\nAt first
glance, all the positive and negative numbers at US sportsbooks can be daunting. The good news is they are easy to read once you understand what you
see.\n\nNegative numbers, such as -110, represent how much money you need to wager to win $100. In many cases, such as a moneyline bet, the negative number also
indicates which side of the bet is the favorite to win.\n\nFor example, if you were to bet on LeBron James scoring over 15.5 points in a contest at -125 odds,
you’d need to wager $125 to win $100. If you did and your bet was correct, then you’d take home $225, which is your original $125 stake returned to you plus
your $100 in profit.', 'NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nPositive numbers, like +145, show
how much you stand to win for every $100 you wager. If you were to bet $50 on Giannis Antetokounmpo to have over 7.5 rebounds and the odds were +130 at the time
you finalized your bet slip, you would win $65 in profit if your bet wins.\n\nPositive numbers also indicate which side of the bet is the underdog in the eyes
of the oddsmakers.\n\nRemember that you can bet as little or as much as you want, and your payout will be calculated based on your wager amount and the odds at
the time you make your bet.\n\nTypes of NBA props\n\nYou can make two main prop bets on the National Basketball Association:\n\nTeam props\n\nPlayer
props\n\nTeam props are typically focused on statistical accomplishments by a particular team and rarely on the game’s result.', 'NBA Prop Bets - Best Team and
Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nYou’ll often come across NBA team prop bets that are based on a team’s points, such as
the first team to reach 20 points, the first team to score in a game, or whether the total points by both teams in a contest will end up being odd or
even.\n\nTeam props, also known as “game props,” don’t rely on individual players but rather on the team as a whole. Player props, on the other hand, are driven
by individual performance.\n\nThese individual props are focused on statistical milestones during a game. The most popular bets typically focus on points,
rebounds, assists, and three-point field goals. You can also bet on combination wagers, such as points and rebounds together.\n\nIn most cases, you can bet on
traditional statistics for the five starters on each team.\n\nSee more:\n\nTips for betting on NBA props\n\nEvery bettor should develop their strategies, but we
do have three tips you can use for NBA props.', 'NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nPay
attention to player and team news: Oddsmakers use various information to develop their lines for games and players, and you should, too. Keep tabs on player
news and injuries, when top players are being rested, how they performed in recent games, and more. You can find value in player and team props by gathering
information from multiple sources and tracking how players and their teammates perform. For example, if one of the starting guards is questionable for a game,
you might want to consider the other guard for increased points or assists since they’ll likely handle a heavier load.\n\nKeep an eye on usage and minutes: Even
slight increases to a player’s average minutes can make a difference when betting player props, especially later in the season when aging veteran players tend
to rest more. Keep yourself informed on usage and minutes trends for the players you’re considering betting on, and use them to your advantage.', 'NBA Prop Bets
- Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)\nRemember that some of the best players in the NBA were once on the bench
for most games until small increases in their playing time gave them a chance to shine.\n\nShop for the best lines: This tip applies to every bet you make.
Using multiple sportsbook accounts lets you find the best odds for every NBA prop bet. Even the smallest differences, like one sportsbook having odds at +125
and another at +135, can make a difference in your profit. By shopping for the best lines, you can make smarter bets and find the best value.\n\nRelated
Pages:']
</item 7>

<item 8>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
They vary from player to team bets and lean heavily on the statistical side of the game. Some NBA prop examples include:

How many points a player will score

Number of assists or three-pointers a player will have

Which player will score the first basket of a game

You can also bet on team props, such as whether a team will lead at the end of a certain quarter or an over/under bet on the total number of points the team
will score in the game.

See more: NBA prop bets

NFL prop bets

The National Football League is the king of sports betting, and NFL betting sites have no qualms about showcasing every kind of possible bet type for it.

Bettors will find many prop bet opportunities, especially for the Super Bowl. During the regular season, one of the most popular prop bet options includes the
“anytime touchdown scorer” bet, where you can wager on a player to score a TD at any point in the game. Other prop bets can be focused on the team or even both
teams.

Graph facts:
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> Super Bowl
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> Super Bowl
- NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 -MENTIONS-> Super Bowl
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Super Bowl
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Super Bowl

Entities: Super Bowl, prop betting, Prop Betting - What is a Prop Bet?, Sporting News, NBA, sports betting, National Football League
</item 8>

<item 9>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
Use most of your budget on more stable bet types, such as moneylines, spreads, or totals. Focus only a small percentage of your bankroll on props.

We wouldn’t recommend using prop bets as your main bet type when building your strategy. Use most of your budget on more stable bet types, such as moneylines,
spreads, or totals. Focus only a small percentage of your bankroll on props. Exotic props are a guessing game: When large events like the Super Bowl roll
around, you’ll be inundated with exotic and entertaining prop bets. These are almost impossible to research, and pure guessing isn’t good for anyone’s bankroll
or betting strategy.

Prop bet examples for major sports

Every major sport with odds at the best sports betting apps will have prop bets. Below, we give examples of prop bets you might find when betting on major
American sports.

NBA prop bets

The NBA offers a wide range of prop bets. They vary from player to team bets and lean heavily on the statistical side of the game.

Graph facts:
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> Super Bowl
- Vermont Sportsbook Promos and Sports Betting Launch News -MENTIONS-> Super Bowl
- NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 -MENTIONS-> Super Bowl
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Super Bowl
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Super Bowl

Entities: Super Bowl, Bankroll, Betting strategy, Exotic props, Prop Betting - What is a Prop Bet?, Sporting News, spreads, totals, NBA, prop bets, moneylines
</item 9>

<item 10>
Prop Betting - What is a Prop Bet? (Sporting News, 2023-11-06T21:23:22+00:00)
Let’s say you’re keeping an eye on the New York Jets vs. the Buffalo Bills, and you see the total shift due to an incoming winter storm. While many bettors will
scramble to get their bet in on the ‘under’ before the lines move too much, it’s a good idea to scope out the prop bets for individual players. The main lines
are shifting, but most of the best sports betting sites don’t have the personnel to keep tabs on every possible player, so you could find value in an ‘under’ on
a quarterback’s passing yards because the lines haven’t caught up yet. There’s greater variety: The three main bet types—moneylines, point spreads, and
totals—are constant in every major sport, but they don’t offer a lot in the way of variety. Prop bets do. They can add interest to a game that may seem
one-sided in the major markets. Larger sporting events, especially the Super Bowl, offer a massive variety of bets, ranging from the typical to the exotic and
everything in between.

Graph facts:
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> New York Jets
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> New York Jets
- NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 -MENTIONS-> New York Jets
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> New York Jets
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> New York Jets

Entities: New York Jets, Super Bowl, winter storm, Prop Betting - What is a Prop Bet?, Sporting News, Buffalo Bills, totals, prop bets, point spreads,
moneylines
</item 10>
=== END ITEM id=9269343ee1e09b9e ===
