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

=== ITEM id=4b12342eee9a4e37 ===
Gold sentences and facts:
Sentence 0: Automakers and technology companies are building ever-more sophisticated digital platforms into the future generations of cars and other vehicles.
  Fact 0.0: Automakers are building ever-more sophisticated digital platforms into future generations of cars and other vehicles.
  Fact 0.1: Technology companies are building ever-more sophisticated digital platforms into future generations of cars and other vehicles.
Sentence 1: Now we have software, robotics, AI and all that stuff, and the car companies need to embrace that.
  Fact 1.0: The speaker says the industry now has software, robotics, AI, and similar technology.
  Fact 1.1: Car companies need to embrace software, robotics, AI, and similar technology.

Retrieved items:
<item 1>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
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
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Looking back on all that had changed around what first attracted people to Google, from the introduction of ads to the efforts to keep users within the universe
of Google products, this seemed to be the last straw.

The second major reason why Google results feel different lately was, of course, SEO — specifically, the obnoxious-kid-refusing-to-behave-in-class kind of SEO.

“SEO that goes against Google’s guidelines, it’s not new,” Ray explained. A decade ago, it used to be called “black hat” SEO, in comparison to the search
engine-approved “white hat” tactics. And Google has, as Sullivan and many SEOs told me, gotten better over time at catching SEOs playing tricks on the
algorithm.
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
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 7>

<item 8>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 8>

<item 9>
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
=== END ITEM id=4b12342eee9a4e37 ===

=== ITEM id=9cdcc419bd9094d2 ===
Gold sentences and facts:
Sentence 0: “The product 6PPD-quinone, which was identified in late 2020, requires more research to understand what impact, if any, it may have on aquatic life or the environment,” Goodyear SVP of global operations and CTO Chris Helsel says.
  Fact 0.0: 6PPD-quinone was identified in late 2020
  Fact 0.1: 6PPD-quinone requires more research to understand what impact, if any, it may have on aquatic life or the environment
  Fact 0.2: Chris Helsel is Goodyear's SVP of global operations and CTO
  Fact 0.3: Chris Helsel says the 6PPD-quinone statement
Sentence 1: In our report, a former VP at an industrial automation vendor shared, “Digital solutions provide the visibility, analysis, and insight needed to address the challenges inherent in sustainability goals.
  Fact 1.0: The report contains a quote from a former VP at an industrial automation vendor
  Fact 1.1: The former VP at an industrial automation vendor said digital solutions provide visibility, analysis, and insight
  Fact 1.2: The digital solutions address the challenges inherent in sustainability goals

Retrieved items:
<item 1>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
These materials are harder to make than MXenes. So most research on them has been done with math or computer models. But MBenes could have unique mechanical and
electrical traits, Ozkan’s team finds. And those just might help them last through many, many cycles of CO 2 capture and release.

Each year, human activities spew tens of billions of tons of CO 2 into the air. That’s equivalent to the weight of tens of billions of elephants. Carbon
absorbers, including MXenes or MBenes, can’t catch it all. As such, they’re never going to replace the need for reducing emissions of CO 2 and other greenhouse
gases.

Still, given the urgency of climate change, even a quick or partial fix — such as carbon capture — can’t be ignored. “At this point in time,” says Persson, “we
need to look at all the possible solutions.” And MXene-like materials are emerging as potentially important candidates.
</item 1>

<item 2>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Based on these observations and additional insights from our report, our current projection estimates that the global digital twin market could grow at a CAGR
of approximately 30% between 2023 and 2027.

Against the backdrop of this growing market, the report notes eight trends that help promote the advancement and adoption of digital twins. Here, we will focus
on four:

1. Digital twins deployed to meet sustainability goals

2. Digital twins employed as virtual sensors in complex conditions

3. Partnerships between cloud hyperscalers and OT and simulation specialists

4. Initiatives promoting interoperability of digital twins across multifaceted systems

Before we dive into these trends, and to provide context, it is helpful to understand how IoT Analytics defines digital twins and the digital twin market.

Defining digital twins and the digital twin market

In our March 2023 Decoding Digital Twins article, we define digital twins as a virtual model replicating the behavior of an existing or a potential real-world
asset, system, or multiple systems.
</item 2>

<item 3>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
It will also promote a green and blue economy coupled with improved governance to guarantee long-term climate and disaster resilience.

BUILDING RESILIENCE THROUGH SUSTAINABLE DEVELOPMENT

Coming off from the pandemic and with the current global economic climate, it is opportune for the government to proactively drive for actionable policies and
programs that focus on building the resiliency of the economy through sustainable development with greater emphasis given to addressing climate change.

Navigating external and domestic economic headwinds will not be an easy feat for the administration, but the private sector will be an important catalyst for
sustainable development. With the private sector sharing industry knowledge, resources and potentially even leading certain socio-economic programs and projects
of the National Government, it is to be hoped that AmBisyon Natin 2040 of long and healthy lives for Filipinos that are strongly rooted, comfortable, and secure
will be achieved.

This article is for general information only and is not a substitute for professional advice where the facts and circumstances warrant. The views and opinions
expressed above are those of the author and do not necessarily represent the views of SGV & Co.
</item 3>

<item 4>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Track tires are pushed to their limits, resulting in better products for the road. In time, at least.

Many manufacturers have a self-imposed deadline for a shift to renewable sourcing for materials. Dunlop is promising 40 percent of its tires will be renewable
by the end of this decade and 100 percent by 2050. Bridgestone’s goal is also 2050 but is introducing models now with an increased focus on sustainability, like
the Turanza EV, made of 50 percent renewable and recycled materials.

How all of this factors into each manufacturer’s specialty blends is complicated, but generally, it breaks down like this:

“Natural rubber and synthetic rubber each represent about 25 percent of materials used in tire manufacturing. Fillers, primarily carbon black and silica,
represent an additional 25 percent,” Bridgestone’s Niaura says. “The remaining percentage consists of steel, textiles, oils, resins, and other ingredients used
to cure the tire or serve as anti-degradants.”
</item 4>

<item 5>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
How worried are you about that?

JC: I think it's great that deep sea mining has made people care more about the deep ocean, but it hasn't actually started yet and research does not support
some of the more hyperbolic headlines.

Related: More than 5,000 new species found in 'pristine' deep-sea wilderness. But they could soon be wiped out.

There's a lot of research focused on how we're going to manage mining, if it does go ahead. And there are some habitat types in the deep ocean that we don't
need to do further research on, because we know they are so vulnerable. We know that we would risk species extinction at active hydrothermal vents, for example,
because they're a tiny habitat globally — just 50 square kilometers [19 square miles] — with more than 400 animal species not found in any other habitat type.
</item 5>

<item 6>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Authoritarians know this, and this is why they have troll farms that can manipulate the range of solutions and the sentiments about them.

The Internet as we experience it represents a central obstacle to climate action, through several mechanisms. Promotion of false information about climate
change is only one of them. There’s general political polarization, which inhibits the coalitions we need to build to realize clean energy, as well as creates
paralyzing infighting within the climate movement about strategies, which the platforms benefit from. There’s networked opposition to the infrastructure we need
for the energy transition. There’s the constant distraction from the climate crisis, in the form of the churning scandals of the day, in an attention economy
where all topics compete for mental energy. And there’s the drain of time and attention spent on these platforms rather than in real-world actions.

Any of these areas are worth spending time on, but this essay focuses on how the contemporary media ecology interferes with climate strategy and infrastructure
in particular.
</item 6>

<item 7>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
In other words: less harmful rubber, more grip, better wear, and better range or fuel economy.

The molecular menace

Firestone’s IndyCar tires.

Drive long enough, and the well-defined tread patterns of even the most efficient tires will fade, slowly balding like a stressed-out environmental scientist’s
pate. It’s easy to imagine that the lost rubber simply disappears, but the truth is more terrifying.

Tires break down into tire and road wear particles, or TRWP. This black dust accumulates on roads mostly — but not permanently. When it rains, it is carried
away.

A recent Pew Charitable Trusts report found that 78 percent of ocean microplastics are synthetic tire rubber, which makes efforts to cut down on microfiber
pollution from our laundry seem a bit quaint.
</item 7>

<item 8>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
“I would wake up in a city and not know what time zone I was in,” he recalled. To slow the pace, he upped it to $5,000 a day, but “it seemed the more I raised
my rates, the more gigs I was getting.”

Nowadays, he mostly invests in cannabis and psychedelics. SEO just got to be too complicated for not enough money, he told me. Ward had told me the same thing,
that she had stopped focusing on SEO years ago.

I was considering how it was possible that so many people have been complaining recently about SEO ruining the internet if these people were telling me the SEO
business is in decline when I met Jairo Bastilla. He was the kind of tall, charming man who described himself multiple times as “a nerd,” and he pointed out
that even though working directly with search engine rankings is “no longer monetizing at the highest payout,” the same “core knowledge of SEO” remains relevant
for everything from native advertising to social media.

Translation? SEO is now baked into everything.
</item 8>

<item 9>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
By the IoT Analytics team.

A new report from IoT Analytics highlights eight notable trends helping to advance and promote digital twins.

Four of these trends are discussed in detail in this article. These trends are shaping the future of the digital twin market and influencing investment
priorities for companies across various industries.

Key insights:

According to our Digital Twin Market Report 2023–2027, the digital twin market is expanding, with a projected CAGR of 30% between 2023 and 2027.

29% of global manufacturing companies have either fully or partially implemented their digital twin strategies. Further, job posts related to digital twins have
increased by 11% compared to October 2021, while openings for other tech topics have declined in the same timeframe.

The report notes eight notable trends helping to advance and promote digital twins, four of which are discussed in this article.
</item 9>

<item 10>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
“We look forward to focusing on the momentum for the business and how we can help the retailers, especially the big retailers who are looking to us to innovate
for them, because this is a tough year for retailers.”

You might remember that Bolt, which provides checkout technology to merchants, and its co-founder Ryan Breslow, were subpoenaed last year by the SEC to
investigate whether the company violated any securities laws during fundraising in 2021 when Bolt was seeking its $355 million Series E round that valued the
company at $11 billion.

The investigation took about 15 months, but news of that probe didn’t go public until July of 2023. Shortly thereafter on August 23, the SEC said, in a letter
viewed by TechCrunch, it was not recommending an enforcement action for the company.

Kuruvilla spoke with me about how, exactly, the company can put something like this behind it, what it told customers and what’s next. Read more.
</item 10>

<item 11>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Right now, the current plot is about the centralization of discourse on a few corporate platforms. The rise of the platforms brings potential to network
democratic uprisings, as well as buoy authoritarian leaders through post-truth memes and algorithms optimized to dish out anger and hatred. This is a more
challenging story to narrate, because the setting is everywhere. The story unfolds in our bedrooms while we should be sleeping or waking up, filling the most
quotidian moments of waiting in line in the grocery store or while in transit. The characters are us, even more intimately than with climate change. It makes it
hard to see the shape and meaning of this story. And while we are increasingly aware of the influence that shifting our media and social lives onto big tech
platforms has on our democracy, less attention is devoted to the influence this has on our ability to respond to climate change.

Think about these three forces meeting — climate change, authoritarianism, the Internet. What comes to mind?
</item 11>

<item 12>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
This will include optimizing the production process of hydrogels as well as sourcing affordable raw materials and minimizing waste generated during the
manufacturing of these materials.

“Ensuring the safety and reliability of hydrogel-based batteries is also necessary,” Fan concluded. “Rigorous testing and evaluation of the potential
environmental impact of hydrogel electrolytes, from raw material extraction to disposal, are vital to meeting sustainability goals and gaining public trust.”

Reference: Peihua Yang, Hong Jin Fan, et al., Hetero-Polyionic Hydrogels Enable Dendrites-Free Aqueous Zn-I2 Batteries with Fast Kinetics, Advanced Materials,
(2023). DOI: 10.1002/adma.202306531

Feature image credit: danilo.alvesd on Unsplash
</item 12>

<item 13>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
In terms of the digital twin market, our market model only considers software spending for digital twins, which we break into two scopes:

Broad digital twin market – includes the revenue for all software solutions that provide capabilities that are used for digital twins, such as solutions that
integrate data sources into a digital twin and are capable of simulation, visualization, and predictions

– includes the revenue for all software solutions that provide capabilities that are used for digital twins, such as solutions that integrate data sources into
a digital twin and are capable of simulation, visualization, and predictions Narrow digital twin market – only considers spending for digital twin-specific
software, such as software intended to model the digital twin data

Though hardware and services related to digital twins can bolster the capabilities of digital twin software, we only view them as supporting services for the
software, and we do not consider them in our digital twin market model. That said, this ability to bolster capabilities means they can aid digital twin market
growth, as shown in some of the trends below.
</item 13>

<item 14>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
“What you don’t want is all these little wear particles full of all these nasty compounds settling in the environment and inside us and gradually leaching out
their toxins,” says Nick Molden, CEO of Emissions Analytics, an independent group focused on testing and data about the overall environmental impacts of
transportation.

“What you don’t want is all these little wear particles full of all these nasty compounds settling in the environment and inside us and gradually leaching out
their toxins.”

One such toxin is 6PPD-quinone. A 2020 study from the University of Washington’s Center for Urban Waters found that a dramatic decline in coho salmon
populations was the result of the toxin. The salmon have now been placed on some West Coast endangered lists.

“Shortly after birth, they were essentially going mad and dying,” Molden says. “The researchers finally linked it to this compound in tires because it was
happening after rainfall. The rainfall was washing all the accumulated tire material off the road into the harbors and into the shore.”
</item 14>

<item 15>
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
</item 15>

<item 16>
The Apple Watch’s double tap gesture points at a new way to use wearables (The Verge, 2023-10-25T17:04:01+00:00)
To my surprise, it feels less gimmicky than I expected. But despite Apple’s efforts, it doesn’t take long to run into double tap’s limitations.

Double tap makes using the Smart Stack more intuitive, but I wish it could scroll AND select. GIF by Amelia Holowaty Krales / The Verge

Media controls are a good example. Should a double tap either pause or play your music? Or should it let you skip to the next track? If you’re someone who uses
your smartwatch to control playback on smart speakers, the former makes more sense. If you’re a runner and the wrong song pops up on your playlist, the latter
is more useful. I’ve run into this in various scenarios, but this happens to be one of only two instances where you can choose what double tap does. (For me,
I’ve set double tap to skip to the next track.)

That lack of choice is also apparent in third-party apps.
</item 16>

<item 17>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
While on his working vacation in Maine, Kalmus invited NBC News for a short visit to the community to see the crisis through the eyes of Americans taking it as
seriously as anyone.

The Possibility Alliance is an 11-acre homestead that operates without power, running water or a sewer system. The Hughes family, which operates the alliance,
pays only one monthly bill for landline phone service. (Evan Bush / NBC News)

Detoxifying

About a decade ago, Kalmus ditched a promising academic career in astrophysics to study the physics of climate change instead.

He made other changes too. He stopped flying, became a vegetarian and ditched gasoline-powered cars (he drives a Tesla), cutting his personal emissions by about
90%, according to his math.

It wasn’t a political statement, but a detoxification.

“Burning fossil fuels started to feel more and more disgusting,” he said.

Ethan Hughes, who operates the Possibility Alliance, rejects globalization and thinks capitalism is a ruinous symptom of “greed and separation.”
</item 17>

<item 18>
Robotics Q&A with Boston Dynamics’ Aaron Saunders (TechCrunch, 2023-12-09T16:05:51+00:00)
When would you pay as much for a robot as you would a car? When it achieves the same level of dependability and value you have come to take for granted in the
amazing machines we use to transport us around the world.

What important robotics story/trend isn’t getting enough coverage?

There is a lot of enthusiasm around AI and its potential to change all industries, including robotics. Although it has a clear role and may unlock domains that
have been relatively static for decades, there is a lot more to a good robotic product than 1’s and 0’s. For AI to achieve the physical embodiment we need to
interact with the world around us, we need to track progress in key technologies like computers, perception sensors, power sources and all the other bits that
make up a full robotic system. The recent pivot in automotive towards electrification and Advanced Driver Assistance Systems (ADAS) is quickly transforming a
massive supply chain. Progress in graphics cards, computers and increasingly sophisticated AI-enabled consumer electronics continues to drive value into
adjacent supply chains.
</item 18>

<item 19>
Creation of quantum dots wins 2023 chemistry Nobel (Science News For Students, 2023-10-04T21:11:57+00:00)
This opened up a world of possible uses for the dots.

For one thing, quantum dots can be used to subtly change the color of LED lights. This can dramatically improve the lights’ energy efficiency.

Quantum dots are also useful in medicine. They can be injected into the body and attached to cells from the body’s immune system. Such cells swarm cancerous
tissues. By tagging the cells with fluorescent quantum dots, surgeons can spot even hard-to-see tumors inside the body.

Quantum dots can also be tuned to absorb different colors of light. So they could be used to build solar panels that soak up sunlight well in different
conditions. The dots may even be used to build quantum computers. Such computers promise to run much faster than any normal computer ever could.

This year’s chemistry Nobel is well-deserved, says Warren Chan. He’s a biomedical engineer and chemist at the University of Toronto in Canada. “They’re the ones
who built the foundation,” Chan says of the winners.
</item 19>

<item 20>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
The partnership addressed similar challenges related to Industry 4.0 and Smart Manufacturing.

DTC digital twin interoperability framework

The Digital Twin Consortium (DTC), a collaborative partnership of industry, academia, and government experts, released the Digital Twin System Interoperability
Framework to “unify a nascent ecosystem of high-value, multi-vendor services that can seamlessly ‘plug into’ a multi-dimensional, interoperable system of
systems.”

DTC and IDTA collaboration

DTC and the Industrial Digital Twin Association (IDTA) announced a liaison agreement to collaborate on standardization requirements, enabling interoperability
through discussions, aligning work in horizontal domains, and collaborating on open-source projects, contributions, and reference implementations.

DTC and OPC Foundation collaboration

DTC and the OPC Foundation announced a liaison agreement to “accelerate the development and adoption of digital twin-enabling technologies,” promoting
interoperability standards and processes to advance the use of digital twins in manufacturing across multiple industries.
</item 20>

<item 21>
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
</item 21>

<item 22>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
It relies on modern composites instead of air pressure to provide the right mix of stiffness and flex with significantly more durability. Michelin says it could
reduce tire replacements by 20 percent. Variations of this theme have been in development for decades, but models are in testing now for consumer and industrial
use.

Goodyear gave us a more radical vision of a more distant future tire with 2020’s ReCharge concept, which proposed that a biodegradable tread could be recharged
simply by inserting refillable capsules with seasonal compounds. As a concept, it’s pretty far out there. Goodyear’s Helsel confesses as much, calling them
“purely conceptual designs” that “might never be produced as such.” However, he says that the non-pneumatic nature of those concepts, and the advanced sensors
embedded within, are indicative of ongoing research into future tires.

Another concept, though, is less about increasing the sustainability of tires and more about increasing the viability of EVs.
</item 22>

<item 23>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
A number of works outline the contours and dynamics of the current media ecology and what it does — Siva Vaidhyanathan’s Antisocial Media, Safiya U. Noble’s
Algorithms of Oppression, Geert Lovink’s Sad by Design, Shoshana Zuboff’s Surveillance Capitalism, Richard Seymour’s The Twittering Machine, Tim Hwang’s
Subprime Attention Crisis, Tressie McMillan Cottom’s writing on how to understand the social relations of Internet technologies through racial capitalism, and
many more. At the same time, there’s reasonable counter-discussion about how many of our problems can really be laid at the feet of social media. The research
on the impacts of social media on political dysfunction, mental health, and society writ large does not paint a neat portrait.
</item 23>

<item 24>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
That isn’t bad just for the salmon and the Pacific Northwest’s fishing industry. 6PPD-quinone is in us, too, and it only comes from tires. 6PPD helps tires
resist environmental damage, thus lasting longer between replacements. When 6PPD reacts with ozone, it becomes 6PPD-quinone.

“The product 6PPD-quinone, which was identified in late 2020, requires more research to understand what impact, if any, it may have on aquatic life or the
environment,” Goodyear SVP of global operations and CTO Chris Helsel says. That’s hardly a mea culpa, but the tire industry genuinely has been an active partner
in studying the problem.

Goodyear and other members of the US Tire Manufacturers Association actually encouraged 6PPD-targeting regulation enacted by the California Environmental
Protection Agency and have provided free tire compound samples to researchers.
</item 24>

<item 25>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Part of / We only get one planet

Tires are the most important safety component on a car, yet there’s increasing research that they’re doing a good deal of harm, too. Slowly, surely, one mote of
black dust at a time, tires are leaving their mark on the environment.

According to researchers at Imperial College London, our cars and trucks emit 6 million tons of tire wear particles every year, transportation detritus that
goes into the air, the land, and the water. That’s a massive figure that ignores the nearly 250 million tires that are disposed of annually in North America,
roughly half of which are burned.

It’s a depressing state of affairs, but there is room for optimism. Every major tire manufacturer globally is raising its game — not only in terms of
performance in the face of bigger, heavier electric cars but also in terms of environmental impact. Renewable materials are finding their way into modern tires,
and there’s even greener rubber on the horizon, too.
</item 25>

<item 26>
Consistent policy is key to unlocking the climate tech boom, but the UK’s U-turn isn’t helping (TechCrunch, 2023-10-13T17:00:20+00:00)
One of the most critical things for startups is research and development (R&D) tax credits. This was a lifeline that gave small, research-intensive companies
months of budget runway every year, as it effectively saw them get a third of the money spent in R&D back. The government announced it would be scrapping that
last year, and it is only thanks to a lobbying effort spearheaded by the Startup Coalition that, at the last minute, a portion of that tax credit was able to be
preserved.

Which leads to the third and most important point: A consistent climate policy. The impact of not having one is playing out in real time, right now. Sunak’s
public U-turn doesn’t just undermine science and play into the hands of the climate skeptics. It also plows headlong into the future of any climate tech company
whose launch plans have to be aligned around decarbonization timelines. This will damage economic confidence. People will lose jobs. And action, alas, will be
further delayed.
</item 26>

<item 27>
Nobel Prize winner shocked by early morning phone call from Sweden: 'It was ringing during the night, but I didn’t answer it because I’m trying to get some
sleep, basically' (Fortune, 2023-10-04T21:15:43+00:00)
But it wasn’t until several decades later that scientists could manufacture quantum dots in a lab.

In the 1980s, Ekimov, 78, and Brus, 80, honed the theory and developed early laboratory techniques for creating particles that emit varying colors by adjusting
sizes. In 1993, Bawendi, 62, developed new chemical methods for producing the particles quickly and uniformly — which soon enabled a variety of scalable
commercial applications, including in electronics displays.

Judy Giordan, president of the American Chemical Society, said she was thrilled at this year’s winners.

“What we care about a lot in chemistry is being able to make and tailor novel structures and architectures to solve problems that help people and the planet,”
Giordan said.
</item 27>

<item 28>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
When water is warmer, the metabolism of things living in the water runs faster and they use up oxygen more quickly, so that makes the problem even worse. And
thirdly, we know that the currents carrying oxygen down to the deep ocean are weakening , because melting ice sheets are making the water fresher and blocking
the formation of dense water than sinks.

Those currents take centuries to complete their journey, which means the changes we have already made are going to carry on being felt for centuries. The deep
ocean is already on track to have 10% less oxygen overall globally than it did in preindustrial times by 2400. It's hard to predict what the knock-on effects
are going to be, but they are going to be widespread and they are coming.

SP: You dedicate much of your time to communicating deep sea science with lay audiences. Why is that so important to you?

JC: I enjoy talking to people about the deep sea because it's not somewhere we think about every day.
</item 28>

<item 29>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Scholars have argued that putting too much emphasis on the platforms can be too simplistic and reeks of technological determinism; they have also pointed out
that cultures like the United States’ and the legacy media have a long history with post-truth. That said, there are certainly dynamics going on that we did not
anticipate, and we don’t seem quite sure what to do with them, even with multiple areas of scholarship in communication, disinformation, and social media and
democracy working on these inquiries for years.

What seems clear is that the Internet is not the connectedness we imagined. The ecology and spirituality of the 1960s, which shaped and structured much of what
we see as energy democracy and the good future today, told us we were all connected. Globally networked — it sounds familiar, like a fevered dream from the
1980s or 1990s, a dream that in turn had its roots in the 1960s and before.
</item 29>

<item 30>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
To understand the dynamic, we need to take a closer look at the concept of energy democracy, as generally understood by the climate movement, and its tenets:
renewable, small-scale systems, and community control. The bitter irony of the current moment is that it’s not just rising authoritarianism that is blocking us
from good futures. It’s also our narrow and warped conceptions of democracy that are trapping us.

This article contains affilate links; if you click such a link and make a purchase, we may earn a commission.
</item 30>

<item 31>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Truly sustainable tires are becoming a reality, but will it be enough?

Renewable rubber

Bridgestone’s “Wireless-In-Wheel” technology.

If you watched any IndyCar road course race this year, basically any of the half-dozen events where the drivers had to turn right as well as left, you might
have noticed some green flashes on those cars’ tires. It wasn’t just a visual flair to an otherwise visually dull part of the equation.

Those tires contained guayule, part of an agrarian prototype effort from Bridgestone, the supplier of spec tires for that series. It’s a high-profile
demonstration of a new domestically sourced, renewable source of rubber that augments other natural sources like that from the hevea tree and even the Russian
dandelion.

Truly sustainable tires are becoming a reality, but will it be enough?

Bridgestone’s Bill Niaura, executive director of sustainable innovation and circular economy, says guayule is a drop-in replacement with many potential
advantages. Chief among them?
</item 31>

<item 32>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
In a new paper published in the journal Advanced Materials, Nanyang Technological University (NTU), School of Physical and Mathematical Sciences professor
Hongjin Fan and his co-authors examine different electrolytes for aqueous zinc ion batteries that take the form of hydrogels — three-dimensional network
structures that are capable of storing large amounts of water — to address some of these challenges.

“The core concept of our research is that we design the hydrogel electrolytes by introducing functional groups [specific groupings of atoms within molecules
that have their own characteristic properties] that carry either positive or negative charges,” Fan said. “Our idea is to use a hetero-hydrogel to mitigate
challenges on both the anode and the cathode sides of the zinc iodine batteries.”

A hydrogel boost

Fan explained that the purpose of this hydrogel modification is to ensure that when zinc accumulates on the anode side, it does so evenly.
</item 32>

<item 33>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
But I'm confident that we will see protection for active hydrothermal vents, because we scientists have been saying that for years.

Hydrothermal vents are some of the most vulnerable deep sea habitats and home to 400 animal species not found anywhere else. (Image credit: Universal History
Archive/Universal Images Group via Getty Images)

SP: Deep sea mining is perhaps more manageable in terms of its impacts than other human activities. If not mining, what is the biggest human threat to the deep
sea?

JC: To my mind, it's climate change . And it affects the deep ocean in lots of different ways. The one that concerns me particularly is deoxygenation — the
reduction in oxygen levels — because deep sea animals need oxygen and they get it from the seawater.

Oxygen is carried down by currents that form in the polar regions and sink and spread throughout the deep ocean. As a result of climate change, the ocean is
getting warmer and that means it can't carry as much dissolved oxygen .
</item 33>

<item 34>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
And most pressing for the communities relying on the coupled models’ climate change projections, To what extent are the remaining carbon budgets [Zhou et al.,
2021] with respect to global warming targets, rates of near-term future warming, and detection and attribution efforts contingent on the pattern effect?

A Collaborative Outlook

Working to solve the problems outlined above at the intersection of different disciplines and with input from different research communities will likely benefit
related research into, for example, the sensitivity of tropical cyclones and future rainfall changes in the southwestern United States or in South America to
surface warming patterns.

Promising new tools and observations are beginning to emerge. They include targeted model experiments and intercomparisons, extended global satellite
observations of clouds and radiation, Argo floats sampling deep-ocean heat uptake, estimates of radiative imbalances before the year 2000, and new observational
constraints on clouds’ sensitivity to environmental controlling factors. These developments encourage an optimistic outlook on our ability to quantify the
pattern effect and its implications over the next couple of years.
</item 34>

<item 35>
Consistent policy is key to unlocking the climate tech boom, but the UK’s U-turn isn’t helping (TechCrunch, 2023-10-13T17:00:20+00:00)
If giants like Ford are feeling the frustration, imagine the mood at a small green tech startup.

So what’s the play? How can the government support those striving against an increasingly insecure (and insincere) backdrop?

The response of any tech company impacted by last week’s news surely has to be this.

First, we need a consistent macroeconomic policy. This has a major impact on startups raising and deploying capital into climate R&D. The global economic
contraction has made it difficult for these businesses to raise any money. Overall approaches to the economy have trickle-down effects on how we, as businesses,
make money and run our operations. Right down to salaries, which alone are hard to keep up with inflation-based pay rises.

The second is a consistent tax policy. One of the most critical things for startups is research and development (R&D) tax credits.
</item 35>

<item 36>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Ray reassured me that I was not crazy. Google results today do feel different from how they felt just five or six years ago for two major reasons. The first was
Google’s response to the disinformation panic around the 2016 election, which involved questioning the notion that the most reliable information could be chosen
by a form of popularity, meaning how many links a site received from other sites. As a result, the algorithm seemed to change its approach to links, especially
when it came to news and sites offering legal, financial, or health advice, and instead paid more attention to what Google came to call E-E-A-T: experience,
expertise, authoritativeness, and trustworthiness.

“E-E-A-T has had a pretty big impact on what types of results you see,” Ray told me. She’s done extensive (and fascinating) research around how certain sites
have fared under these new guidelines: Urban Dictionary, down! Mayo Clinic, up! Some people consider EEAT part of what’s making results better than ever.
</item 36>

<item 37>
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
</item 37>

<item 38>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Kent has done his fair share of SEO jobs but also has something of an outsider’s perspective. For years, he’s been telling people that part of the SEO
industry’s reputation problem is that 80 percent of SEOs are scammers.

“A lot of companies and individuals out there selling their services as SEO gurus don’t know what they’re doing or don’t really give a damn,” he explained. As a
consultant, he’s often had businesses ask him to vet the work of other SEOs. “I would take a look at their site and determine the firm had done next to nothing
and had been charging thousands a month for years on end.”

When I ran this 80 percent scam figure by other SEOs, most agreed it sounded accurate, though people were divided about what to ascribe to greed and what was
just stupidity.

“It isn’t because they have a scammer’s heart,” said Bruce Clay. “It’s because they don’t have the real expertise.”
</item 38>

<item 39>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Forrester has a holistic understanding of the delicate push and pull between the SEOs desperate for clues on how to do their jobs better and the search engine
trying to keep its secret-sauce algorithm proprietary. He also knows a huge range of people in the industry. Like Schwartz, he wanted to emphasize how hard
everyone works. “I’ve lost track of how many people I know who built companies and sold them and have just, like, made wealth,” he told me. “That is not a
40-hour commitment in the week. That is a 400-hour commitment.” (For the record, there are 168 hours in a week.)

These days, Forrester lives in Los Angeles, and he asked me to meet him at one of his favorite restaurants, which felt like a British pub operated by Disney
World, tucked away in a desert strip mall. Inside, every inch was covered in Anglophile paraphernalia, including Union Jack flags, a mural of Big Ben, and a red
phone booth.
</item 39>

<item 40>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Some people consider EEAT part of what’s making results better than ever. Others see it as a form of censorship, disproportionately affecting right-wing
perspectives. Not every search query takes EEAT into account; Google has described heightened concern over sites that could impact safety, happiness, and the
ability to be an informed citizen. But the point that really hit me was that for certain kinds of information, Google had undone one of the fundamental elements
of what had made its results so appealing from the start. Now, instead of wild-west crowdsourcing, search was often reinforcing institutional authority.

You can’t just be the most powerful observer in the world for two decades and not deeply warp what you are looking at

This felt complicated at best. When it comes to health and wellness, for example, quackery is often in the eye of the beholder. Everyone knows someone who has
struggled with the limits of Western medicine. So much of the original draw of the internet was the opportunity for outlier voices to be heard alongside
established experts and elites.
</item 40>

<item 41>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Too often, design is minimized to only the act of making user interfaces pretty. Product design is much deeper and broader in scope than visual design assets.
For example, product design can give direction and focus to a business strategy, user experience strategy, or technology explorations.

The process establishes a guide rail throughout any innovation initiative. The core of product design is decision-making with an astute instinct for making the
best decisions at the most opportune time. Product design helps reduce risk and leads to more effective innovation through quality decision-making.

A progressive role for technologists

Technologists play a strategic role in product innovation and should bring a metaphysical perspective in addition to being punctilious. Our job is to
communicate the essence of a technology and think strategically about the application of technologies to problem spaces. We are most constructive when we
translate the technicals of “how to make X do Y” to “these are the types of products and services realizable with technology X.”
</item 41>

<item 42>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Digital twin market snapshot

Data from our Digital Twin Market Report 2023–2027 indicates that 29% of global manufacturing companies have either fully or partially implemented their digital
twin strategies, marking a noticeable increase from 20% in 2020. Moreover, the proportion of these companies not contemplating the implementation of digital
twins appears to have reduced to 9% in 2023, down from 33.6% in 2020.

Supporting this trend, an analysis of job postings on SimplyHired suggests a growing demand for digital twin expertise. Postings mentioning “digital twin”
increased by 11% in October 2023 compared to just two years prior, while openings for most other tech topics declined in the same timeframe. Notably, among the
60 tech-related skill sets we tracked, digital twin-related skills experienced one of the most significant growths.
</item 42>

<item 43>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
Noel P. Rabaja the strategy and transactions (SaT) service leader of SGV & Co.
</item 43>

<item 44>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
This definition captures a macro concept of digital twins, but our report and previous article present a three-dimensional, cuboid model to help classify
digital twins so apple-to-apple comparisons can be made. Each axis of the cuboid represents one dimension of a digital twin:

1. Life cycle phase: The X-axis represents the six life cycle phases a digital twin is used for, from design to decommissioning.

2. Hierarchical levels: The Y-axis represents a digital twin’s five hierarchical levels, from information to multi-system.

3. Use/purpose of implementation: The Z-axis represents the seven most common uses for digital twins, such as simulation and prediction.

Based on this model, there are 210 potential combinations (5 x 6 x 7 = 210); however, our research indicates that many digital twin initiatives are tailored to
multiple combinations.
</item 44>

<item 45>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.
</item 45>

<item 46>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
He has been on a 15-year quest to halt climate change and has already oriented most everything in his life to minimize his own climate footprint and maximize
what he can do to push society to change.

Nothing has been enough.

And so this summer, as extreme heat and rainfall battered many American communities, Kalmus visited Maine searching for something different. He was looking not
for climate solutions, but for new ways to withstand climate change’s harms — and also to make peace with the grief that often feels inescapable for those who
reckon with global warming daily.

climate scientist exercise bicycle environmentalist (Evan Bush / NBC News)

That search brought him to an unusual community called the Possibility Alliance, a homestead where a small group of people is trying to figure out how to live
in a world headed for a breakdown (as they see it).

Its leaders eschew capitalism, grow much of their own food, and live without electricity. They’ve dedicated their lives to protest, living off the land as
simply as possible, and are preparing their community for ecological disaster and to host climate refugees.
</item 46>

<item 47>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
The stock would have been easily missed if PE was the metric used to spot the performers.

Hence EV-based metrics are options to consider as well when equity-based valuation multiples do not give a clear picture. Where even EBITDA multiples are not
clear, as was the case with many new-age company IPOs that were reporting losses even at the EBITDA level, investors have no option but to go further up the P&L
to metrics like EV/revenue, or even outside of P&L to use metrics like EV/subscriber.

As we mentioned earlier, relative valuation represents a number backed by a credible and well-validated story. Hence, you can use non-P&L multiples. So when you
compare companies and invest in one based on EV/Subscriber, there must be a credible story backing future monetisation of the subscriber in a profitable manner.
But do bear in mind, the risks are higher and so may be the rewards or pain as the case may be. When the risks are higher, you must apply a higher margin of
safety before choosing the stock.
</item 47>

<item 48>
Earnings week ahead: DocuSign, NIO, Broadcom, GameStop and more (Seeking Alpha, 2023-12-03T13:02:00+00:00)
The Seeking Alpha's Quant Ratings moved to a Hold recommendation just days before the earnings announcement. Meanwhile, Wall Street analysts maintain a Buy
rating on the stock.

Individual Trader, a Seeking Alpha author, suggests that MOMO's recent downtrend may be concluding, pointing to signs of a bottom forming. The analyst added
that the combination of cheap earnings and a high dividend yield makes it an appealing investment option.

Consensus EPS Estimates: $0.35

Consensus Revenue Estimates: $413.43M

Earnings Insight: The company has a track record of beating EPS and revenue estimates in 7 of the past 8 quarters.

Also reporting: Johnson Outdoors (JOUT), Canadian Western Bank (OTCPK:CBWBF), and more.
</item 48>

<item 49>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Trend 4: Initiatives promoting interoperability of digital twins across systems from different vendors

Agreement on integration standards among the various digital twin technology providers is crucial for building cross-system digital twins. With standards,
manufacturers can offer services that can be applied to unique situations with other digital twin technology and software.

Recognizing this, countries and industry organizations have taken several steps to address this need:

Plattform Industrie 4.0 and CESMII

Plattform Industrie 4.0 and the Clean Energy Smart Manufacturing Innovation Institute (CESMII) partnered to establish standards for manufacturers in Germany and
the United States to better enable smart, sustainable competition. The partnership addressed similar challenges related to Industry 4.0 and Smart Manufacturing.
</item 49>

<item 50>
Check Point sees 'fantastic' year ahead even as rival report disappointing billings (Seeking Alpha, 2023-11-18T13:56:00+00:00)
Palo Alto (PANW) said it nonetheless sees demand for its products.

Earlier this month, Fortinet (NASDAQ:FTNT) got a slew of analyst downgrades over a disappointing fourth-quarter outlook.

The near- to medium-term is uncertain after disappointing results, Cantor Fitzgerald said. Stifel, which likes the company, said moving parts will take time to
play out and shares are expected trade sideways in coming quarters.

The future

Check Point (CHKP) has made three recent acquisitions to expand its offerings.

It has also invested in R&D and launched new products, like the Rugged Industrial Control Systems Gateway, which provides artificial intelligence security and
deployments in harsh environments. Other new offerings include Playblocks, which extends the reach of siloed security systems to stop attacks across products,
people and processes.
</item 50>

<item 51>
X plans to use payment, phone and ID verification to stop bots, in addition to the new $1/yr fee (TechCrunch, 2023-10-18T17:33:48+00:00)
A report this week from market intelligence firm Similarweb, for example, found that X’s web traffic and monthly active users on mobile had declined
post-acquisition.

Users today have far more options for microblogging and public discourse, thanks to products from Twitter-like startups and tech companies such as Spill,
Bluesky, Pebble, Countersocial, Spoutible, Hive and the open source platform Mastodon, as well as efforts from tech giants, like Meta’s new app Threads. Few of
the other social apps currently charge a fee, unless it’s for premium features, like some of the third-party Mastodon apps do.

In addition to presenting friction at sign-up, other critics have argued a fee could exclude genuine users and increase the digital divide.
</item 51>

<item 52>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Trend 3: Partnerships forged in the clouds: Hyperscalers team up with OT and simulation specialists

In the last few years, cloud hyperscalers like AWS and Microsoft Azure have introduced digital twin platforms, e.g., AWS IoT TwinMaker and Azure Digital Twins,
that allow for interconnecting various data sources and building digital twin topology. However, these companies realize they cannot deliver end-to-end digital
twin solutions by themselves and are thus partnering with OT and simulation companies to add capabilities to their networks.

Hyperscalers and OT companies

OT companies provide industrial data management capabilities and connections to millions of local physical assets. By partnering with hyperscalers, OT companies
can reach broader audiences looking for digital transformation solutions by offering typical cloud benefits like storage and computation power.

Example: In 2021, Siemens and AWS announced an expansion of their partnership, with digital twin technology being an area of focus.
</item 52>

<item 53>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
For the energy sector in particular, amendments to the PSA are envisioned to help raise the capital needed to speed up the energy transition of the country to
renewables.

In addition, the Regional Comprehensive Economic Partnership (RCEP), a trading bloc that encompasses the ten members of the Association of Southeast Asian
Nations (ASEAN) and the ASEAN Plus One Free Trade Agreement (FTA) partners Japan, China, South Korea, Australia, and New Zealand, is already in force for the
Philippines, helping ease market access through trade and investment rules and supporting global and regional supply chains. The Philippines can potentially
position itself as a regional manufacturing hub if the right incentives and policy measures are put in place to encourage local and foreign investors to
participate. Together, trade and investment are expected to play an important role in attaining economic transformation, the second goal of the PDP.

HUMAN CAPITAL AND INFRASTRUCTURE

The ability of people to take advantage of market opportunities arising from investment rests on human capital.
</item 53>

<item 54>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
What comes to mind? If you recombine the familiar characters from these stories, perhaps it looks like climate activists using the capabilities of the Internet
to further both networked protest and energy democracy. In particular, advocacy for a version of “energy democracy” that looks like wind, water, and solar;
decentralized systems; and local community control of energy.

In this essay, I would like to suggest that this is not actually where the three forces of rising authoritarianism x climate change x tech platforms domination
leads. Rather, the political economy of online media has boxed us into a social landscape wherein both the political consensus and the infrastructure we need
for the energy transition is impossible to build. The current configuration of the Internet is a key obstacle to climate action.

The possibilities of climate action exist within a media ecosystem that has monetized our attention and that profits from our hate and division. Algorithms that
reap advertising profits from maximizing time-on-site have figured out that what keeps us clicking is anger.
</item 54>

<item 55>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
The bad news is that we’re still going through more tires than ever. Harrigle says we scrap 250 million tires every year in North America. Again, about half are
burned for fuel.

But there are other solutions. Recycling is one, but reuse is better, a process called retreading. Retreading itself is nothing new, but while it’s generally
fallen out of favor for consumer tires, on the industrial side of things, it can be a huge cost-saver — and environment-saver, too.

Jason Roanhouse, VP of North American Bandag operations at Bridgestone Mobility Solutions, says that the company’s retread business has saved 300 million tires
from being thrown out, worth roughly 4 billion gallons of oil.

Future tires

While efforts to find sustainable replacements for tire manufacturing compounds continue, as does research into what today’s compounds do to our health, many
manufacturers are trying to change the equation altogether.

A near-future example is the Michelin Uptis, a non-pneumatic tire.
</item 55>

<item 56>
Discogs’ vibrant vinyl community is shattering (The Verge, 2023-10-07T14:00:00+00:00)
According to the sellers who spoke with The Verge, bugs abound: the European seller, for example, had just been dealing with an issue with the platform’s refund
button. “Discogs said it was PayPal’s fault, and PayPal said it was Discogs’ fault,” he says. “It caused stress for the buyers, and so I had to do direct
refunds — which meant I was refunding not just what I made but Discogs’ and PayPal’s commissions as well, effectively losing money on the refund.” Melkisethian,
speaking a month later, had just noticed a shift in the way shipments are processed that required manually entering information in steps that used to be
automated.

Besides the baseline functionality of the site, there are other improvements that could bring Discogs closer in line with its competitors. “There are other
seller tools and seller initiatives that we’ve been asking for for years that have never been done — like any kind of tie-in with Google, any kind of
integration with social media, the kinds of things basically other platform has,” says Martin.
</item 56>

<item 57>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
So water and oxygen make MXenes less reactive. But scientists think they can overcome this problem by tweaking MXenes’ structure. Capping the materials’ edges
with bulky salts, for instance, could shield MXenes from damage by oxygen.

It’s also hard, right now, to make large amounts of MXenes. In the lab, researchers can make only several kilograms (pounds) in a single batch. Several
companies are working out how to boost batch sizes.

But Thakur points out one irony: Making MXenes for carbon capture can have a big carbon footprint. It takes a lot of electricity to get the heat needed to make
them, she notes. Much of that electricity is made using CO 2 -spewing fossil fuels. It’s important to make sure that the energy used to make these materials
doesn’t spew more CO 2 than the MXenes can later sop up. Otherwise, that would defeat their use in carbon capture.
</item 57>

<item 58>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
The latest funding round brings QI Tech’s total capital raised to $262 million.

New Web Summit CEO appointed: Former Wikimedia Foundation CEO Katherine Maher has been appointed as the new CEO of Web Summit, following the controversial
departure of its previous leader, Paddy Cosgrave. Maher’s appointment comes as the tech conference faces a challenge to rebuild relationships and trust within
the tech community.

Top reads on TechCrunch this week

Shinier, smarter maps: Google is rolling out a variety of updates for Maps, incorporating AI technology to enhance user experience. The feature is currently
available in France, Germany, Japan, the U.K. and the U.S., with plans for further expansion.

From stop-and-start to charging ahead: Over on TC+, Tim has had it with U.S. automakers hesitating on EVs, as Ford and GM are showing signs of backing away from
their commitment to electric vehicles. Cowards.
</item 58>

<item 59>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
The government calls for embedding resilience, sustainability, and nature-based solutions in infrastructure planning and investment to enhance climate
resilience. Likewise, investments in renewable energy are expected not only to enhance energy security but also reduce greenhouse gas (GHG) emissions.

From an economic vantage point, GHG emissions are economic “bads” and are a cause of inefficiency as they get over-produced. Their effects, however, go beyond
national borders, and while the Philippines contributes minimally to global GHG emissions, the World Risk Index 2022 report identifies the Philippines as the
most disaster-prone country in the world.

The government envisions that by 2028: (1) Climate and disaster risk resilience of communities and institutions will increase (2) Ecosystem resilience will be
enhanced, and (3) A low-carbon economy transition is enabled.

To ensure that these are realized, the government is set to safeguard cross-sectoral convergence and implement a comprehensive risk management approach to
address the adverse consequences posed by climate change.
</item 59>

<item 60>
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
</item 60>

<item 61>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
The polyiodides are immobilized at the cathode side of the battery, and reactions in which iodine is both stripped of electrons (oxidization) and also granted
electrons (reduction) — so-called “redox reactions” — are sped up thanks to the synchronized rapid transport of ions and electrons.

The solid nature of the hydrogel electrolyte can also limit the growth of dendrites thanks to its mechanical strength, which acts as a physical barrier not
present in liquid electrolytes.

Tentative first steps

Testing their aqueous hydrogel-modified zinc iodine batteries battery, the team found that the decay rate was ultralow, with the energy storage capacity of the
battery retained to an outstanding level over the course of 18,000 discharge and recharge cycles. This outstrips the stability of currently available zinc
iodine batteries, and their test battery also delivered some surprising results.
</item 61>

<item 62>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
One of the major drawbacks of these lithium-ion batteries is the inherent risk of burning or even explosion as the result of thermal runaway, with overheated
lithium-ion batteries deemed responsible for a number of fires in the past.

A possibility with zinc

One potential substitute for lithium-ion batteries are rechargeable zinc iodine batteries , which have the advantage of being safer, costing less, and being
more friendly to the environment than lithium alternatives.

However, zinc iodine batteries are not perfect. Composed of an negative iodine electrode and a positive zinc electrode , these batteries suffer from issues like
lower energy output compared to commercial non-aqueous lithium batteries and the growth of zinc dendrites on the anodes — projections of material that build on
electrodes and short batteries when they cross from one electrode to another — as well as the corrosion of the zinc anode that result in reduced efficiency and
even short-circuiting.
</item 62>

<item 63>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
When the risks are higher, you must apply a higher margin of safety before choosing the stock.

Another thing to bear in mind when it comes to EV-based valuation — any change in EV will impact only the equity value. For example, if EV of a company is ₹100
and it has ₹50 in debt, then the value of its equity is ₹50. In this case, if you think EV can increase by 10 per cent, the value of equity will increase by 20
per cent, and the reverse is also true. The higher the leverage, more the impact of change in EV on the value of the stock (both ways).

Keep a weather eye on Ketchup Economics

Given the fact that relative valuation is based on comparison with peer multiples, investors must take measures to check whether the peer multiples are
reasonable as well. An overvalued peer cannot be used to justify expensive valuation in one stock. This is what economist Larry Summers termed as Ketchup
economists/economics.
</item 63>

<item 64>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Months later, after returning “kicking and screaming” to the “quote-unquote real world,” Kalmus was planning a return trip to the Possibility Alliance over the
winter holidays.

He was still mulling why the Possibility Alliance’s “low energy” lifestyle had felt so meaningful and what was missing from modern life.

Ethan Hughes his goats out to

“Everything is too fast,” Kalmus said. “It’s obvious to a lot of people, but it’s a new revelation to me. There’s nothing more important in life than
relationships.”

This article was originally published on NBCNews.com
</item 64>

<item 65>
Consistent policy is key to unlocking the climate tech boom, but the UK’s U-turn isn’t helping (TechCrunch, 2023-10-13T17:00:20+00:00)
When British prime minister Rishi Sunak appeared in front of the hastily assembled press on September 20, the letter-crammed slogan on his lectern caused the
country to squint: “Long-term decisions for a brighter future,” it read.

We now know, of course, there was little in the speech that followed that brought hope. Certainly not concerning the technological fight for our future climate.

Not that long ago, the UK seemed a rather brighter beacon in the industrial transition toward reversing the global climate breakdown. The countrywide goals were
laid out. COP26 at least offered a forum and a spotlight. London has made strides in establishing itself as a hub for green tech startups. On the narrow but
viable path toward net zero, leaders were at least taking the right steps.

Then came the nadir of the last few weeks.
</item 65>

<item 66>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
It was easy: buy links that led to your site and watch your ranking in Google’s results rise.

I came to understand that, since the dawn of the internet, there have been people attempting to manipulate search and then people decrying those manipulations
as the end of search’s ability to be useful. It works in cycles. People doing SEO find loopholes in the algorithm; critics complain about search results; search
engines innovate and close the loopholes. Rinse, repeat.

Before our current moment of widespread disillusionment with online information, the rise of SEO had reached a breaking point multiple times. In 2003, as Google
approached the deadline to disclose pertinent business information leading up to its IPO, the company quietly released an update cracking down. By 2011, SEO was
once again oppressively pervasive. TechCrunch published a story called “Why We Desperately Need a New (and Better) Google,” which argued that “Google has become
a jungle: a tropical paradise for spammers and marketers.”
</item 66>

<item 67>
Meltwater, the media monitoring startup, gets a $65M investment from Verdane (TechCrunch, 2023-12-18T10:54:31+00:00)
In that context, it’s notable that Verdane opted to invest in Fountain Venture rather than directly in Meltwater. That will give Verdane not only the stake in
Meltwater, but also a stake in Jobylon and whatever else Fountain and Lyseggen find interesting. That will, in turn, de-leverage a focus on just one business.
Verdane itself has only recently started to spread its wings to invest in startups across all of Europe and beyond: tying up with a partner to help direct it is
a very de-risked approach to take while trying to be more ambitious.

In terms of technology, companies like Meltwater are at a crossroads these days. The company would have had its roots out of the businesses where humans would
have physically sifted through piles of newspapers, daily, to clip mentions of a company’s name, collate those and send them on to those clients to help them
better track how they are being covered in the media.
</item 67>

<item 68>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Battery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software that can be slipped into just about any
lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.

What lies beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched
on where startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.
</item 68>

<item 69>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
New research aims to improve the stability and safety of alternatives to rechargeable lithium-ion batteries using aqueous zinc and hydrogels.

The demand for rechargeable batteries as a result of the growth of both the personal electronics and electric vehicle markets has grown steadily over the last
decade. Lithium-ion batteries are one of the technologies that have emerged and matured due to this demand.

In 2020, Statista estimated that lithium-ion batteries accounted for 185 gigawatts (GWh) of energy consumed, and this energy demand is projected to reach 2,035
GWh by 2030. Most of this growth is accounted for by the increased desire for electric vehicles as people turn away from fossil fuels.

While it is undeniable that lithium-ion batteries are more environmentally friendly than fossil fuels and while they are also relatively inexpensive, these
batteries aren’t perfect.
</item 69>

<item 70>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
He speculated that the company didn’t want the public to think too hard about how search works because that awareness “encourages regulators and makes people
realize, ‘Oh, there’s a lot of money here.’”

There has always been advertising and polemics from cranks, scammers, and liars. But now we see this stuff surfacing alongside truth, and we can’t tell the
difference. We move through our lives with a greater sense of distrust and fear and insecurity. At the end of the day, it’s Google’s world, and the SEOs are
only living in it.

And as much as I might hate the way the SEOs who don’t follow Google’s rules have altered my online experience, the reality is that most people running a
company will break whatever rules they are able to get away with breaking.
</item 70>

<item 71>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
If not, what investments are required?

2. Data integration: How compatible are our current data sources and systems with digital twin technologies, especially in terms of sensor data (both physical
and virtual/soft)?

3. Interoperability concerns: As we use multiple software solutions across our operations, how important is interoperability between digital twin technologies
and our existing systems?
</item 71>

<item 72>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
The thing we’re jacked into and collectively creating seems more like a global endocrine system than anything we might have visualized in the years while
“cyber” was a prefix. This may seem a banal observation, given that Marshall McLuhan was talking about the global nervous system more than fifty years ago. We
had enthusiasm about cybernetics and global connectivity over the decades and, more recently, a revitalization of theory about networks and kinship and rhizomes
and all the rest. (The irony is that with fifty years of talk on “systems thinking,” we still have responses to things like COVID-19 or climate that are almost
antithetical to considering interconnected systems — dominated by one set of expertise and failing to incorporate the social sciences and humanities). So —
globally connected, yet divided into silos, camps, echo-chambers, and so on. Social media platforms are acting as agents, structuring our interactions and our
spaces for dialogue and solution-building.
</item 72>

<item 73>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
Welcome back to The Interchange, where we take a look at the hottest fintech news of the previous week. If you want to receive The Interchange directly in your
inbox every Sunday, head here to sign up! After a bit of a quiet period, things in the world of fintech picked up in a big way this past week. We wrote about
some notable fundraises, how PayPal was hit with an antitrust lawsuit, how Bolt is trying to move on after an SEC probe, and much more.

Rainforest takes on the bigger players

It’s not every day that we’re pitched companies that feel like they are taking on bigger players in a really meaningful way. This past week, I wrote about
Rainforest, an Atlanta-based startup that is taking on incumbents such as Fiserv and FIS, as well as trying to take market share from other fintechs such as
Stripe with its offering. Rainforest works with software companies to help them embed financial services and payments into their platforms.
</item 73>

<item 74>
Earnings week ahead: DocuSign, NIO, Broadcom, GameStop and more (Seeking Alpha, 2023-12-03T13:02:00+00:00)
Also reporting: Science Applications International Corporation (SAIC), Fusion Fuel Green PLC (HTOO), IDT Corporation (IDT), JOANN (JOAN), RGC Resources(RGCO),
and more.

Tuesday, December 5

NIO (NIO)

NIO (NIO) is all set to release its Q3 earnings before the bell on Tuesday. The Wall Street consensus remains at Buy, while Seeking Alpha’s Quant Ratings
maintain a Hold rating on the stock due to concerns about profitability and valuation.

On Friday, the EV maker announced that deliveries rose 12.6% Y/Y but dipped roughly 1% M/M to 15,959 vehicles in November.

Stone Fox Capital, an SA Investing Group leader, notes that NIO's stock has fallen below $8, attributing the move to fluctuating monthly delivery numbers and
inconsistent EV deliveries. Despite achieving an electric vehicle delivery record in Q3, the recent 10% workforce reduction raises concerns for Q4, Stone Fox
Capital noted.
</item 74>

<item 75>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Considerations for digital twin vendors

3 questions digital twin vendors should ask themselves based on research findings discussed in this article:

1. Sustainability goals: Given the emphasis on sustainability and the role of digital twins in achieving it, is my company’s digital twin solution well-equipped
to support sustainability objectives? Are there new features we need to develop?

2. Interoperability: Is our digital twin solution currently designed to be interoperable with other systems and technologies, especially those from different
vendors?

3. Product development: How aligned is our product development roadmap with the emerging trends highlighted in the report, especially around end-user and
technological advancements?

Considerations for adopters

3 questions adopters should ask themselves based on research findings discussed in this article:

1. Infrastructure readiness: Do we have the necessary infrastructure and resources in place to support the deployment and effective use of digital twins? If
not, what investments are required?

2.
</item 75>

<item 76>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
The companies aimed to accelerate Siemens Xcelerator adoption and democratize new digital twin solutions using AWS IoT TwinMaker, a service intended to make
creating digital twins that incorporate multiple data sources faster and easier.

Hyperscalers and simulation companies

Simulation companies bring specialized expertise in creating accurate, high-fidelity models of physical entities, be it machinery, buildings, or entire
ecosystems. By partnering with a hyperscaler, simulation companies can harness these cloud services’ data storage and computational power to create more robust,
responsive, and accurate data twins that can be scaled and integrated seamlessly into broader IT ecosystems.

Example: In 2023, Ansys and Microsoft announced a partnership to help customers envision digital twins on a large scale. In cooperation with Tata Consultancy
Services, Ansys and Microsoft integrated Ansys’s Twin Builder’s physics-based simulation capabilities with Azure Digital Twin and IoT data to allow Twin Builder
users to run scenarios and obtain predictions on how their systems will behave.
</item 76>

<item 77>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
This has the effect of mitigating the diffusion back and forth of polysulfide between anode and cathode, the so-called “shuttle effect”, which can cause poor
cycling stability and severe anode corrosion in batteries. The use of a hydrogel electrolyte also has the effect of accelerating the iodine conversion reaction
on the cathode side.

Fan explained that the anode and cathode of the hydrogel have different functional groups that make it both zincophilic (with an affinity for zinc ) and
iodophilic ( with an affinity for iodine), respectively.

On the anode side of the battery, this has the effect of repelling dissolved polyiodides — negatively charged ions composed entirely of iodine atoms — allowing
the zinc to be deposited uniformly, protecting the anode from iodine-caused corrosion.
</item 77>

<item 78>
Check Point sees 'fantastic' year ahead even as rival report disappointing billings (Seeking Alpha, 2023-11-18T13:56:00+00:00)
The financial damage from cybercrimes will reach an astronomical $10.5 trillion by 2025. As global economies go, that would rank third behind only the U.S. and
China.

But what’s bad for global conglomerates and the average person could be an opportunity for cybersecurity firms like Check Point Software (NASDAQ:CHKP).

Even as rivals like Palo Alto Networks (PANW) and Fortinet (FTNT) cite an uncertain macroeconomic environment for weak billings forecasts, the need for
cybersecurity is clearly growing. Global cybercrime damage costs are expected to increase 15% annually over the next two years, according to Cybersecurity
Ventures.

“There’s never been a better time to be in cyber,” Check Point (CHKP) President Rupal Shah Hollenbeck said in a Zoom interview this week. “It’s fortunate in
that organizations are understanding the business value of cybersecurity in a really big way. Unfortunately, it’s because there are more hacks out there than
ever before.”
</item 78>

<item 79>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
“For the next five years, if you want to sell tires in California, you need to go through the process of either eliminating 6PPD, or at least properly
researching the alternatives,” Molden says.

“We are collaborating with researchers and other scientists to better understand this transformation product, fill knowledge gaps, and determine next steps,”
says Helsel.

Other initiatives are coming to limit overall tire emissions, including the upcoming Euro 7 regulations, which put caps on tire wear particles as well as other
emissions.

That seems like progress, but Emissions Analytics’ Molden isn’t so sure: “The reason I’m worried about that is I’m sure you can reduce that wear mass, but if
you do that by making the compound more toxic, you could actually make the environmental impact worse.”

“You could actually make the environmental impact worse.”

That could be akin to increasing a car’s fuel efficiency by switching to leaded fuel.

Molden believes that what’s needed is better understanding.
</item 79>

<item 80>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
Certainly most of the R&D budget inside of Toyota is spent figuring out how to make the next car or maybe the car five years from now. But there’s this notion —
and I’m sure Hyundai shares this — that we’re at the once-in-a-century time for transformation in the car industry. Battery electric vehicles are much simpler
to make than cars have been in the past — no engine, no transmission, etc. So, we’re going to have to compete in a much fiercer way in cars. But also, can we
use the skills, the dreams and the hopes of the company to go beyond cars? TRI’s job is to actually think about [what’s] next. What’s next after cars, as well
as some fancy stuff on cars.”

For TRI, much of the “what’s next” focuses on supporting aging populations. The organization invests a good portion of its resources to building out technology
designed to help older people live more independently. That’s at the root of the research we shared from the institute last week.
</item 80>

<item 81>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
“The adhesion between the polyanionic and polycationic hydrogel components is remarkably robust, effectively mitigating the swelling of the hydrogel even in the
absence of a cellulose paper scaffold,” Fan explained. “The unexpectedly strong iodine adsorption capability of the positively charged hydrogel has led to a
striking and rapid color transformation, exceeding our initial expectations.

Despite promising results, Fan acknowledged there is much to do before the battery can be brought to the mass market.

“We have to note that in terms of mass or volume energy density, so far, the ZIB is still far from the commercial lithium-ion batteries,” Fan conceded. “The
hydrogel electrolytes need to be further improved in terms of ion conductivity and stability. In addition, hydrogel electrolytes should be compatible with other
battery components, including electrodes and separators.”

The researcher also pointed out that large-scale and cost-effective production methods for hydrogel electrolytes are a prerequisite before the aqueous zinc
iodine batteries is ready for widescale commercial use.
</item 81>

<item 82>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
To this end, the first goal of the PDP is social transformation which includes, among others, sustained expenditure on social services, mainly education,
health, and social protection.

The first two goals of the PDP help attain the developmental objectives of efficiency and equity, which rest on the pillars of sustainability and resilience.
The third fosters societal resilience: an enabling environment encompassing institutions, macroeconomic stability, and the physical and natural environment.

Institutions are vital to economic acceleration, which is why the government’s steps to enhance the ease of doing business are most welcome. Infrastructure
development also enables an economy to sustain higher levels of growth, which, in turn, catalyzes yet more investment. The Public-Private Partnership (PPP) Act
is up for the signature of President Marcos and, if approved, is expected to further enhance the business atmosphere in mobilizing private resources for
infrastructure development.

The government aims to sustain its infrastructure program at 5-6% of GDP through 2028 amid a six-year medium-term fiscal framework.
</item 82>

<item 83>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
Notable reads and other tidbits

Autonomous vehicles

Waymo driverless vehicles are now available through the Uber app, starting with Phoenix. The launch comes five months since the two companies announced a
multi-year agreement for the autonomous vehicle service to be accessed via the Uber app.

Electric vehicles, batteries & charging

Ford is delaying about $12 billion in planned investments on EVs, including construction of a second battery plant with joint venture partner SK On due to
softening demand for higher-priced premium electric vehicles. While EV sales have grown, consumers aren’t willing to pay a premium for an EV over a gas or
hybrid vehicle. That price pressure has squeezed profits, and in the case of Ford’s EV business caused losses to grow.

General Motors and long-time partner Honda have ended plans to build millions of affordable and smaller electric vehicles as the automakers come to terms with
high interest rates and battery costs coupled with softening EV demand. Anyone spotting a trend here?
</item 83>

<item 84>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
Public policies attempt to fulfill one or more of the following objectives: efficiency, equity, and stability. Efficiency and stability are necessary conditions
for sustainable growth while equity helps make development inclusive.

This is the second article in our series following the 2nd SGV Tax Symposium, which focused on how a sustainable and effective tax ecosystem can advance the
sustainability agenda for both the public and private sectors.

In this article, we will discuss the Philippine strategy for sustainable development.

GROWTH THROUGH TRADE AND INVESTMENT

The Philippine Development Plan (PDP) 2023-2028 aims for a deep economic and social transformation to reinvigorate job creation and accelerate poverty reduction
by steering the economy back to a high-growth path. In this regard, two main performance indicators are identified and will be monitored by the National
Government.

The first is for the country to graduate into upper middle-income class status within the term of the current administration.
</item 84>

<item 85>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Even worse, the system is addictive, with notifications delivering hits of dopamine in a part of what historian and addiction expert David Courtwright calls
“limbic capitalism.” Society has more or less sleepwalked into this outrage-industrial complex without having a real analytic framework for understanding it.
The tech platforms and some research groups or think tanks offer up “misinformation” or “disinformation” as the framework, which present the problem as if the
problem is bad content poisoning the well, rather than the structure itself being rotten. As Evgeny Morozov has quipped, “Post-truth is to digital capitalism
what pollution is to fossil capitalism — a by-product of operations.”
</item 85>

<item 86>
Reliance Industries: With the stock having underperformed Nifty 50 in last three years, is it time to buy? (Business Line, 2023-12-09T13:15:42+00:00)
On these lines, RIL has identified green energy as its next growth opportunity. As the world moves away from conventional fossil-based energy sources, towards
cleaner and sustainable energy sources, the company is looking to capitalise on the huge demand potential for renewable power anticipated over the next few
decades.

The company has committed a whopping ₹75,000 crore investment into building a comprehensive green energy and green material ecosystem. At Jamnagar, it is
building a green energy complex spread over 5,000 acres of land which, according to the company, will be among the largest integrated renewable energy
manufacturing facilities globally. This will cost it ₹60,000 crore, while the balance ₹15,000 crore is being invested in partnerships, building future
technologies, value chain to be able to build an integrated renewable energy ecosystem. The company has already made several investments in companies that are
working on green energy technologies — such as battery technologies, solar cells/panels, clean mobility solutions, software tools for solar energy, etc.
</item 86>

<item 87>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Kalmus, trailed by a reporter, buzzed around the homestead like a giddy kid, hiking back and forth to a home in an eco-village near the Possibility Alliance
where he was living and working on his research. He tramped down a forest trail in bare feet, paused work for a midday dip in a shady stream and never changed
his T-shirt.

He ground his coffee with a mortar and pestle, thumbed through a book of Enneagram personality types on the porch with his son Zane and listened to Ethan’s
stories about sailing across the Atlantic in a cargo ship to avoid the carbon emissions of flying.

“I feel met by Peter,” Sarah Wilcox-Hughes said of Kalmus. “I don’t feel like there are many people prioritizing the needs of the planet as he is and we are.”

Kalmus found life in Belfast, and at the Possibility Alliance, slower, richer and more connected to the natural world.
</item 87>

<item 88>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Technologists must “find the grain” of a technology. In this way, they become an intermediary between the abstract nature of design and the pedantic nature of
technology. This philosophical perspective is especially important when a product is in a growth stage or when using emerging technologies.

Regardless of whether your product is growing or in a stable stage or employs established or emerging technologies, integrating technologists into the product
strategy and design process enriches the final outcome. There is a technology perspective extending beyond the operations and mechanics of the code “factory
floor,” which provokes innovation. Sometimes this leads to small, impactful moments of innovation — and sometimes it is a brilliant revolution.
</item 88>

<item 89>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Digital twin market trends

We will now look at four trends helping digital twin market growth.

Trend 1: Digital twins deployed to meet sustainability goals

Sustainability was discussed in approximately 21% of recent CEO earnings calls and has remained a consistent topic throughout 2023. This pairs with our report’s
analysis that the pursuit of sustainability goals is a tailwind macro factor for the digital twin market.

To achieve their sustainability goals, many companies are exploring digital twins. Due to their ability to simulate real-world conditions and deliver real-time
information, organizations can optimize resource usage, reducing carbon emissions and improving supply and transportation networks.

According to Capgemini Research Institute’s 2022 digital twins report, 57% of organizations agreed that one of the key drivers for their digital twin
investments was improving their sustainability, and 51% agreed that digital twins would help achieve their organization’s environmental sustainability goals.

In our report, a former VP at an industrial automation vendor shared, “Digital solutions provide the visibility, analysis, and insight needed to address the
challenges inherent in sustainability goals.
</item 89>

<item 90>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
workforce, but they’re already radically redefining the meaning of work.

Gen Zers want a sense of purpose, so they prioritize environmental, social, and governance (ESG) in the workplace, which encompasses their companies taking
accountability for their sustainability and environmental impacts, providing education and awareness for social issues, and ensuring diverse and inclusive
boards and teams.

And it tracks: Roughly two-thirds of Gen Zers say they frequently speak about important societal issues while at work, according to the Edelman report. They’re
also influencing their older coworkers when it comes to areas like work-life balance, fair pay, and employers’ involvement on social issues.

Gen Z’s financial stress

But don’t forget, Gen Zers have also lived through a global pandemic that shuttered the world for nearly two years, two recessions, and a mounting student debt
crisis, leaving them with little savings but an abundance of financial despair.
</item 90>

<item 91>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
I mean, have I even mentioned that they call themselves “SEOs”? Really. They say things like, “As the SEO, my job is to get more traffic.” This title feels
thirsty to be seen as similar to a CEO, to be taken seriously. And compared to the rest of the tech world, SEO has always lacked a certain glamor or a certain
messiah complex. Case in point: while many of the tech CEOs claiming to save the world these days live in Miami, the alligator party was an hour up the coast in
Fort Lauderdale.

“The SEO people are just trying to make money,” said Peter Kent, the author of several dozen explanatory tech books, including SEO for Dummies and Bitcoin for
Dummies. “The cryptocurrency people are trying to make money, but they’re also trying to overthrow, you know, the existing system.”

Kent has done his fair share of SEO jobs but also has something of an outsider’s perspective.
</item 91>

<item 92>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
A digital twin strategy as part of an overall digitalization plan can be a crucial capability for asset-intensive industries and needs to encompass the entire
asset lifecycle, process, and value chain from design and operations through maintenance and strategic business planning.”

Example: Aden, a Chinese integrated facility service provider, created a digital twin for one of its commercial centers in Chengdu, China. The digital twin was
designed to help facility managers inspect, maintain, and repair building assets. 3D simulations assist facility managers in visualizing, predicting, and
optimizing energy consumption, and the expected benefits include lower annual energy consumption, water usage, and waste.

Trend 2: Digital twins employed as virtual sensors in complex conditions

Virtual sensors approximate data that otherwise cannot be obtained via physical sensors, often due to physical sensors being impractical, costly, or hazardous
to employ. Companies are now building digital twins to model hardware and calculate data based on other conditions. This not only allows the collection of data
from complex equipment but also enables operators to track equipment performance and predict maintenance and downtimes.
</item 92>

<item 93>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Molden believes that what’s needed is better understanding. To that end, he’s recently created the Tyre Emissions Research Consortium, a LinkedIn group with
about 1,000 members. The intent is to break down the silos that individual researchers were working in.

Research into 6PPD replacements is ongoing, and the US Tire Manufacturers Association is also touting research into various ways to mitigate and reduce tire
debris pollution.

Given all that research is happening at an industry level, it’s easy to feel a little helpless about the role we’re all playing in contributing this pollution
to the environment when we’re just trying to get to work. Still, there are things that we can do to help minimize the environmental impact in the interim.

Molden suggests a few things: drive more slowly, drive more smoothly, and choose the right tires. According to Molden, the tires that your car was outfitted
with when new will provide the best wear characteristics.
</item 93>

<item 94>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Google is now making it the default sign-in method for all users.

2 sec, let me text you some cash: When questioned about Mastercard’s prospects in emerging markets such as India, Mastercard’s CFO Sachin Mehra praised UPI for
helping with digitization but voiced reservations about its commercial sustainability, saying it is an “incredibly painful experience” for ecosystem
participants.

Bravely browsing — or searching — for a new job . . . : Brave Software, the maker of Brave Browser and Search, confirmed that it has laid off 9% of its
workforce across departments.
</item 94>

<item 95>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Chief among them? Location: “Hevea trees are primarily grown in Southeast Asia,” he says, “while guayule is grown in the Southwestern US.”

This is also a climate resiliency play. Guayule provides an alternate drought-resistant source for natural rubber at a time when global temperature patterns are
behaving like the stock market: highly unpredictable but trending upward.

Virtually every tire manufacturer is making at least token efforts to increase the usage of sustainable materials like guayule. Yokohama is using motorsports to
develop its Advan A005 tire, which debuted at last year’s Pikes Peak International Hill Climb and contains 33 percent sustainable materials, including
biomass-derived rubber. Michelin, meanwhile, shod Porsche’s prototype electric GT4 e-Performance racer with tires said to be 53 percent renewable, like strap
steel and lemon peels.

That all may sound like marketing plays — and it partly is — but motorsports have historically proven an ideal development opportunity in the automotive
industry.
</item 95>

<item 96>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
It won’t always be right, though. But it becomes the epicenter of where the conversation is. And then I’ll think what percentage of our population really wants
to earnestly take the breath and say, let me stop doomscrolling, and let me just wait until people can do the work, synthesize it and then help me react?” he
continued.

“We are also so plugged into all these storylines, whether it’s from traditional media, whether it’s from user-generated — someone on their phone, you know,
posting to a Telegram group. We are overwhelmed and overloaded, and it’s unfortunately appealing to the worst parts of a lot of our instincts,” Ohanian added.

In part, some of the issues around how social media platforms developed had to do with how the companies were originally developed. Ohanian, for example, shared
that Reddit grew out of his running of a forum site in college, when he believed that online community felt as real as offline.
</item 96>

<item 97>
Consistent policy is key to unlocking the climate tech boom, but the UK’s U-turn isn’t helping (TechCrunch, 2023-10-13T17:00:20+00:00)
because it offers the best possible place to make their vision happen.” Unless his long-term vision was only meant to last until the end of the summer, for
entrepreneurs, innovators and businesses to thrive and unlock the economic potential that comes from creating new industries, we need a consistent approach from
the government.

We have enormous fiscal potential. Within the U.K. climate tech community, we are working to create high-paying jobs and value for investors across just about
every asset class. And collectively — hell, individually — our solutions could genuinely change the world.

Of course, this is the common cause we should be united around. Our company is trying to move industry away from a reliance on fossil fuels to make chemicals,
among other things. But the impact of these political spasms and contradictions in our climate commitments has on, say, an EV battery business — which has just
seen the market demand for its output slide down the road by five years — isn’t hard to appreciate.
</item 97>

<item 98>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Example: In large motor applications, such as room-sized motors that pump high volumes of gas, oil, or other chemicals, operators need to monitor the motors’
temperatures—especially if the motors are high-powered and repeatedly started. If a motor is too hot during a restart, the components could cause serious
damage. Unfortunately, using a direct, physical sensor within a motor is often impractical, and operators must work from an assumptive time it takes for the
motor to cool (with an additional safety buffer time). This means motors can be down longer than necessary, impacting efficiency and revenue.

To address this issue, Siemens developed a prototype virtual sensor based on a digital twin. It offers a simulation of how a physical sensor would operate if it
were possible to install it within a motor. Using AR headsets, operators can see a simulation of the motor and its interior with a demonstrator superimposed
over it, and they can see the motor’s temperature.
</item 98>

<item 99>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Media theorist Geert Lovink reflects on a 1996 interview with John Perry Barlow, Electronic Frontier Foundation cofounder and Grateful Dead lyricist, in which
Barlow was describing how cyberspace was connecting each and every synapse of all citizens on the planet. As Lovink writes, “Apart from the so-called last
billion we’re there now. This is what we can all agree on. The corona crisis is the first Event in World History where the internet doesn’t merely play ‘a role’
— the Event coincides with the Net. There’s a deep irony to this. The virus and the network ... sigh, that’s an old trope, right?” Indeed, read through one
cultural history, it seems obvious that we would reach this point of being globally networked, and that the Internet would not just “play a role” in global
events like COVID-19 or climate change, but shape them.
</item 99>

<item 100>
Creation of quantum dots wins 2023 chemistry Nobel (Science News For Students, 2023-10-04T21:11:57+00:00)
“They’re the ones who built the foundation,” Chan says of the winners. “I’m really happy that the field is getting credit for really changing the world.”

Chan and his colleagues found one of the first uses for quantum dots in the 1990s. Their group used quantum dots to tag cells in lab experiments. But the Nobel
committee doesn’t just look at past impacts of a discovery, Chan notes. They also consider the effects a discovery may have in the future. And quantum dots
could have a whole medley of uses not yet explored. Chan’s team, for instance, is now using quantum dots to detect infections such as flu and HIV.

“I was absolutely thrilled to see this,” says Judith Giordan. She’s the president of the American Chemical Society. “We have three people recognized who brought
this technology from a dream, a hope, a theoretical construct … all the way through synthesis and manufacture.”
</item 100>
=== END ITEM id=9cdcc419bd9094d2 ===
