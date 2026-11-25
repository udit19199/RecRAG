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

=== ITEM id=4122a721d322f321 ===
Gold sentences and facts:
Sentence 0: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 0.0: In the last roundup, it was reported that Google spent $26.3 billion in 2021 making itself the default search engine across platforms.
  Fact 0.1: In the last roundup, it was reported that Google tried to have Chrome preinstalled on iPhones.
Sentence 1: When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.
  Fact 1.0: Google's search head Prabhakar Raghavan testified in court on October 28.
  Fact 1.1: Prabhakar Raghavan revealed in his testimony that Google paid $26.3 billion in 2021.
  Fact 1.2: Google paid the $26.3 billion in 2021 to multiple browsers, phones and platforms.
  Fact 1.3: Google's 2021 payments included payments to companies including Apple, Samsung and Mozilla.
  Fact 1.4: The Verge reports this account.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content, their readers and ad revenue.
  Fact 2.2: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 2>

<item 3>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Or maybe sacrifice his firstborn?

The algorithm is just too opaque, too complicated, and too dynamic, making it easy for scammy SEOs to pretend they know what they’re doing and difficult for
outsiders to sort the good SEOs from the bad. To make things even more confusing for, say, a small business looking to hire someone to improve their Google
ranking, even a talented SEO might need a year of work to make a difference, perhaps implying a good SEO was a scammer when in fact, the client was just being
impatient or refusing to implement essential advice. “There’s a great deal of effort that’s required to do things to move the needle, and a lot of companies
aren’t willing to put out the money for that, even though it may be worthwhile in the long run,” said John Heard, a longtime SEO based in Kansas.

Of course, some people bristled at the very suggestion that the industry is filled with con artists. “There are a lot of scammers in every single business.
</item 3>

<item 4>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 4>

<item 5>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
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
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
prompts that get around the safety filters in place to attempt to prevent it from discussing controversial topics.

Using an automated method to algorithmically change the context of prompts until Gemini Pro’s guardrails failed, AI security researchers at Robust Intelligence,
a startup selling model-auditing tools, managed to get Gemini Pro to suggest ways to steal from a charity and assassinate a high-profile individual (albeit with
“nanobots” — admittedly not the most realistic weapon of choice).

Now, Gemini Pro isn’t the most capable version of Gemini — that model, Gemini Ultra, is set to launch sometime next year in Bard and other products. Google
compared the performance of Gemini Pro to GPT-4’s predecessor, GPT-3.5, a model that’s around a year old.

But Google nevertheless promised improvements in reasoning, planning and understanding with Gemini Pro over the previous model powering Bard, claiming Gemini
Pro was better at summarizing content, brainstorming and writing. Clearly, it has some work to do in those departments.
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 8>

<item 9>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
To help address misinformation, Google has also announced that it will soon be integrating new innovations in watermarking, metadata, and other techniques into
its latest generative models.

“Google also recently joined other leading AI companies to jointly commit to advancing responsible practices in the development of artificial intelligence which
will support efforts by the G7, the OECD, and national governments. Going forward we will continue to report and expand upon Google developed AI tools and are
committed to advance bold and responsible AI, to maximise AI’s benefits and minimise its risks.”

Over the next six months Google’s report states it has no additional measures planned for YouTube. But, with generative image capabilities rolling out
internally over the next year, it commits Google Search to leveraging IPTC Photo Metadata Standard to add metadata tags to images that are generated by Google
AI.

“Creators and publishers will be able to add a similar markup to their own images, so a label can be displayed in Search to indicate the images as AI
generated,” Google’s report further notes.
</item 9>

<item 10>
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
</item 10>

<item 11>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
A comprehensive list of 2023 tech layoffs From major layoffs at Google, Amazon and Microsoft to small fintech startups and apps

Last year’s techwide reckoning continues. The tech industry has seen more than 240,000 jobs lost in 2023, a total that’s already 50% higher than last year and
growing. Earlier this year, mass workforce reductions were driven by the biggest names in tech like Google, Amazon, Microsoft, Yahoo, Meta and Zoom. Startups
across many sectors also announced cutbacks through the first half of the year. And while tech layoffs slowed down in the summer and fall, it appears that cuts
are ramping up yet again.

Many economists have cautioned against fears of a recession, which would seem like a reason for optimism. But the momentum for a tech sector rebound has been
slow to build, resulting in tech companies continuing to cut back on their workforces and pivot from a growth mindset to one based on efficiency in the face of
stubborn market conditions.
</item 11>

<item 12>
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
</item 12>

<item 13>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
2. Do they disagree on long-term strategy?

Despite being the hottest tech company in the world right now and everyone talking about ChatGPT, OpenAI isn’t exactly a sound business. It’s shoveling money
into the furnace as fast as it can by serving, by all accounts, a fantastically expensive product at bargain-bin prices.

That’s all well and good for a year or two, but at some point that strategy changes from a growth hack to an existential liability. Could Altman and the board
have had irreconcilable differences on where that point lies?

This doesn’t seem so likely. The company has been very deliberately pursuing this very publicly, confidently and on a long-term basis. Altman and the board seem
to be in sync on this, at least for the present.

3. Do the numbers not add up?

On the other hand, could OpenAI be losing even more money than Altman admitted or projected? It seems impossible, but the costs of running this operation have
no precedent, nor really does the operation itself.
</item 13>

<item 14>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 14>

<item 15>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Although many of us may have rosy memories of how magical and cool Google seemed in the early days, most SEOs consider the years between 2003 and 2011 to be the
boom times, when you could still get a fake corporate website listed above the real corporate website, and you could mess with the search results for a major
political figure such that something sexual or racist would come up first.

Google is harder to game now — it’s true. But the sheer volume of SEO bait being produced is so massive and so complex that Google is overwhelmed. “It’s
exponentially worse,” Ray said. “People can mass auto-generate content with AI and other tools,” she went on, and “in many cases, Google’s algorithms take a
minute to catch onto it.”

The future that Babin had cackled about at the alligator party was already here. We humans and our pedestrian questions were getting caught up in a war of
robots fighting robots, of Google’s algorithms trying to find and stop the AI-enabled sites programmed by SEOs from infecting our internet experience.
</item 15>

<item 16>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
However, the startup wants subscriptions to be its primary revenue driver. Deft is currently testing two paid tiers: a $10 a month plan, which will offer better
search customization to filter out some brands, and a $20 per month plan, which will enable personal shopping service, that will continue to look for suitable
products even after you stop searching manually.

Up next, Deft aims to release a browser extension that helps users search for products by the end of this year or early next year. In the next three years, the
company wants to build and release search APIs that other e-commerce sites could use.

Challenges of building a search startup

Given that it’s building a search solution for e-commerce, Deft already faces challenges from Amazon and Google. In September, Amazon updated its visual search
feature that lets users snap a photo of a product and search for similar items on the platform. Google released its multi-search feature last year that lets
users fuse images and text.
</item 16>

<item 17>
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
</item 17>

<item 18>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.
</item 18>

<item 19>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
You can’t just be the most powerful observer in the world for two decades and not deeply warp what you are looking at.

For the past 25 years, the internet as we know it has been almost entirely defined and controlled by Google. What the SEOs do matters for all of us on a daily
basis, distorting how we perceive the world in ways we can hardly begin to imagine or understand. Yet any money that any SEO has made is a fraction of a crumb
compared to Google’s 10-layer cake. The company brings in hundreds of billions of dollars a year, profits that skew Google’s choices and priorities. As Google’s
founders wrote back in 1997: “we expect that advertising funded search engines will be inherently biased towards the advertisers and away from the needs of the
consumers.”

At the end of the day, it’s Google’s world, and the SEOs are only living in it

There’s a reason why most countries around the world have libraries that are public institutions: information that is controlled by a private business will
always be subject to that business’s bottom line.
</item 19>

<item 20>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
</item 20>

<item 21>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“We try to get people up and running as fast as possible.”

Cue acknowledged that the ISA didn’t allow Apple to offer users a choice of search engines during setup but also said he wouldn’t want to do that anyway. “We
try to get people up and running as fast as possible,” he said. “Setup is just critical stuff.” Showing people a bunch of search engines they’ve never heard of
would just be a bad user experience, he argued; even Cue couldn’t remember the names of some of the alternatives to Google. “We make Google be the default
search engine,” he said, “because we’ve always thought it was the best. We pick the best one and let users easily change it.” (“Easily” is a persistent point of
contention in this trial — DuckDuckGo’s CEO, who testified last week, claimed it takes “too many steps” to switch.)

As for the privacy pop-ups? This is where Bellshaw began to press on how exactly Apple decided Google had the best product.
</item 21>

<item 22>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 22>

<item 23>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
When I finally manage to jump through the flaming rings necessary to be allowed to speak on the phone with Sullivan, albeit with a communications chaperone also
on the line, I find him angry and defensive. He’s annoyed that anyone would think his era at Google has been less transparent than Cutts’ was: “We have reams of
help documents!” he told me. “We have more people assigned to work with SEOs than we did when Matt worked here!”

Sullivan is mad that the public and the media don’t really understand what he considers to be basic precepts about how search works, leading him to adopt a
rather scolding tone online. He’s frustrated that people want to know every last detail about Google’s algorithm because even “if we listed all one thousand of
the ranking signals” and how much each was worth, he said, that wouldn’t actually help SEOs do their jobs better, anyway.

And most of all, Sullivan is pissed that people think Google results have gone downhill. Because they haven’t, he insisted.
</item 23>

<item 24>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
He realized that most search engines were just listing websites in order of how many times a search term appeared on the site and in its tags, so he focused on
stuffing his sites with keywords, resubmitting his URL to the search engines, and waiting for the results to change.

Mapstead started pulling in $25,000–$30,000 a month, working 12- to 14-hour days. “It was how long could I stay awake and how little life could I have because
this was more money than I could have ever imagined in my lifetime,” he told me. “It was like I won the lottery, and I didn’t know how long it would last.”

Around this time, in 1997, an Italian professor published a journal article about what he called Search Engines Persuasion. “Finding the right information on
the World Wide Web is becoming a fundamental problem,” he wrote. “A vast number of new companies was born just to make customer Web pages as visible as
possible,” which “has led to a bad performance degradation of search engines.”

Enter Google.
</item 24>

<item 25>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
</item 25>

<item 26>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
the list of blue links in search results that are not sponsored. “Ad keyword matching is a long-standing and well-known process that is designed to connect
people to relevant ads. A separate process, which has nothing to do with ads, is used to match organic results to a query,” he said. “It’s no secret that Google
Search looks beyond the specific words in a query to better understand their meaning ... If you make a spelling mistake, or search for a term that’s not on a
page but where the page has a close synonym, or if you aren’t even sure exactly how to search for something, our meaning matching systems help.” It is true that
some search queries result in more ads than others, and what ads you get can vary wildly depending on your request, which is the result of the keyword matching
Google uses to sell ads. But Google denies pushing users towards the most monetisable subjects.
</item 26>

<item 27>
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
</item 27>

<item 28>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Over the phone, Cutts came off as humble and thoughtful, acknowledging the nuances and challenges of the search engine business, while Sullivan sounded like an
impatient corporate stooge, trying to gaslight me into believing the sky was red.

But here’s the part where I started to feel the way I’ve felt so often in recent years, like I was losing my grip on reality: Sullivan was not the only person
who tried to tell me that search results have improved significantly. Out of the dozen-plus SEOs that I spoke with at length, nearly every single one insisted
that search results are way better than they used to be. And except for Sullivan, these were not people with an incentive to praise Google. If anything, these
were folks who lamented how much harder it had become for them to take advantage of Google. Today, they told me, search results are just objectively more
accurate. More useful. More difficult to manipulate.

This was not what I had been noticing, and this was certainly not what I had been hearing from friends and journalists and friends who are journalists.
</item 28>

<item 29>
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
</item 29>

<item 30>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
</item 30>

<item 31>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Cutts, as an individual, seemed to be doing his best within an expanding corporate behemoth to remain approachable. “One thing I learned early on was that even
when someone was shouting at you, there’s a kernel of something you needed to hear in the other person and listen to and respect and integrate and incorporate,”
he told me. Most SEOs told me they appreciated his efforts. When Google released the 2011 Panda update that devastated a generation of SEO businesses, Cutts
openly recognized the impossible task of achieving the kind of epistemological neutrality that Google’s founders had initially promised, telling Wired at the
time, “[T]he only way to be neutral is either to randomize the links or to do it alphabetically.”

Still, some blamed him personally for “killing” companies that had relied on the previous iteration of the algorithm. During his time at Google, Cutts regularly
received death threats and hate mail.
</item 31>

<item 32>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Some people consider EEAT part of what’s making results better than ever. Others see it as a form of censorship, disproportionately affecting right-wing
perspectives. Not every search query takes EEAT into account; Google has described heightened concern over sites that could impact safety, happiness, and the
ability to be an informed citizen. But the point that really hit me was that for certain kinds of information, Google had undone one of the fundamental elements
of what had made its results so appealing from the start. Now, instead of wild-west crowdsourcing, search was often reinforcing institutional authority.

You can’t just be the most powerful observer in the world for two decades and not deeply warp what you are looking at

This felt complicated at best. When it comes to health and wellness, for example, quackery is often in the eye of the beholder. Everyone knows someone who has
struggled with the limits of Western medicine. So much of the original draw of the internet was the opportunity for outlier voices to be heard alongside
established experts and elites.
</item 32>

<item 33>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 33>

<item 34>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Looking back on all that had changed around what first attracted people to Google, from the introduction of ads to the efforts to keep users within the universe
of Google products, this seemed to be the last straw.

The second major reason why Google results feel different lately was, of course, SEO — specifically, the obnoxious-kid-refusing-to-behave-in-class kind of SEO.

“SEO that goes against Google’s guidelines, it’s not new,” Ray explained. A decade ago, it used to be called “black hat” SEO, in comparison to the search
engine-approved “white hat” tactics. And Google has, as Sullivan and many SEOs told me, gotten better over time at catching SEOs playing tricks on the
algorithm.
</item 34>

<item 35>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Ray seemed like the most reasonable person I had spoken to so far. Sure, she called herself a “thought leader,” and yes, sure, she had changed her last name to
improve her personal branding by more closely associating herself with her grandmother’s uncle, the artist Man Ray. Maybe some people would say that’s the kind
of absurd behavior that merits being attacked by an alligator, but I was beginning to come down on the side of the SEOs, who seemed to have a lot less agency
than I’d first imagined.

Google had started with a noble cause: trying to make the internet easier to navigate at scale. The company did accomplish that goal, but in doing so, it
inadvertently and profoundly changed how the internet looked. The problem lay in Google trying to be an objective and neutral arbiter of an information
landscape that was meant to pretend it did not exist. You cannot design a free, automated system to help people find information without some people trying to
game that system.
</item 35>

<item 36>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“Before 2003,” Cue said, “the way that you searched the web was you had to go in and you had to type in google.com in the URL field, or you could type in
another URL. We came up with the idea that if you type anything in the URL field that’s not a URL, it just goes to search.”

Schmidtlein’s overall point was that Google helped Safari succeed not by forcing Apple’s hand, but by being a great product that integrated seamlessly with
Apple’s own stuff. He referenced Apple’s deals with Yahoo and Bing that make those services easy to find, and both men argued that switching search engines is
so easy as to be a non-issue. Bellshaw briefly stepped up to rebut that notion, and that was it for Cue’s testimony.

At least, that’s all the testimony we saw. Like so many things in this trial, the star witness was kept mostly under wraps thanks to complaints and worries
about revealing confidential numbers and corporate secrets.
</item 36>

<item 37>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Google, which is also seeking to capitalise on surging demand for AI technology, saw revenue climb 11 per cent to $US76.7 billion, with net income of $US19.7
billion thanks to a recovery in digital advertising revenue. However, its closely watched cloud division, which houses its data storage business and many of its
AI efforts, missed Wall Street expectations. “Investors were disappointed by the relatively weak performance at its Google cloud platform, which is at risk of
falling further behind” competitors’ offerings, Jesse Cohen, a senior analyst at Investing.com, wrote in an email. Loading During trading hours, the S&P 500
slipped off session highs, led by losses in energy shares as oil slid below $US84 a barrel, but once again the index found support above the key 4200 mark.
Verizon Communications, 3M and General Electric climbed on bullish forecasts. Facebook and Instagram owner Meta dropped after being sued by California and dozen
of other US states over harmful youth marketing claims.
</item 37>

<item 38>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Despite cloud computing remaining critical, AI budgets are now more than double that of cloud computing. Eighty-two percent of business leaders believe the
integration of AI solutions will increase their employee performance and job satisfaction, and startups should expect a high level of desire for and experience
with AI solutions in their future customers.

Finally, we’ve seen consumption, or usage-based priced tech products’ growth slow in recent quarters. Will that fact lead startups building modern AI tools to
pursue more traditional SaaS pricing? (The OpenAI pricing schema based on tokens and usage led us to this question.)

The trajectory of usage-based pricing has organically aligned with the needs of large language models, given that there is significant variation in
prompt/output sizes and resource utilization per user. OpenAI itself racks upward of $700,000 per day on compute, so to achieve profitability, these operation
costs need to be allocated effectively.

Nevertheless, we’ve seen the sentiment that tying all costs to volume is generally unpopular with end users, who prefer predictable systems that allow them to
budget more effectively.
</item 38>

<item 39>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 39>

<item 40>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Eventually, a site filled with computer-generated nonsense designed to maximize SEO will get removed from search results, Ray explained, but while it’s up, the
creator might make as much as $50,000 or $100,000 a month. A lot of the people who did this, she said, live cheaply overseas in places like Bali and Chiang Mai.
”They make a bunch of money, that site dies, and they go do it again,” she said. “It’s like a churn and burn strategy. So if people are seeing those results, it
can be very frustrating for users ‘cause it’s like, ‘This is terrible.’”

And yet, as much as she despises what this kind of SEO has done to the internet, Ray told me she hesitated to condemn the actual people doing it. “I used to do
those types of tactics, so I couldn’t hate on anybody personally,” she said. “If people have a problem with Google’s results, they have to ask themselves, is it
the fault of the SEOs?” she asked.
</item 40>

<item 41>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 41>

<item 42>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 42>

<item 43>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 43>

<item 44>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
As the price passed $US30 million, Microsoft quit, but it rejoined the bidding at $US37 million. “We felt like we were in a movie,” Hinton said. Then Microsoft
dropped out a second time. Only Baidu and Google were left, and they pushed the bidding to $US42 million, $US43 million. Finally, at $US44 million, Hinton and
his students stopped the auction. The bids were still climbing, but they wanted to work for Google. And the money was staggering. It was an unmistakable sign
that deep-pocketed companies were determined to buy the most talented AI researchers, which was not lost on Hassabis at DeepMind. He had always told his
employees that DeepMind would remain an independent company. That was, he believed, the best way to ensure its technology didn’t turn into something dangerous.
But as Big Tech entered the talent race, he decided he had no choice: It was time to sell.
</item 44>

<item 45>
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
</item 45>

<item 46>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
During his time at Google, Cutts regularly received death threats and hate mail. When SEOs would send, say, a fruit plate or a brownie cake addressed to him at
Google’s offices, he told me, “We’d take it down to the kitchen with a note warning: possibly poisoned.”

After Cutts left, Google replaced him with a handful of people, none of whom could quite fill his shoes: “Those personalities sometimes were standoffish,”
Forrester told me. “Some of them were superior. Some of them were a bit too wallflower.”

One of the people Google brought in was Danny Sullivan, a former journalist who started Search Engine Land, the industry publication where Schwartz works, back
in the 2000s.
</item 46>

<item 47>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 47>

<item 48>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
He said that the main problem for startups in this space is the go-to-market strategy and fundraising.

“The problem with competing with e-commerce players is not just about user experience, it’s also about logistics infrastructure and supply chain. So a startup
catering to consumer search has to invest a lot of money into making it a viable business,” Konigsberg told TechCrunch.

He also added that it will be difficult for a startup like Deft to build user loyalty while competing against Google and Amazon.

Both Pienaar and Konigsberg said that Deft will have to involve retailers in the mix either as partners or as customers of their search product. Either way,
it’s going to be interesting to see if Deft can build a user base on its own or sell its search feature to e-commerce retailers.
</item 48>

<item 49>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Before Rogier launched MasterClass, he worked as a VC, and — through his connections — he received a $500,000 seed round before he even had an idea for a
company.

And on Chain Reaction, Jacquelyn interviewed David Pakman, managing partner and head of venture investments at CoinFund. Before CoinFund, David spent 14 years
at the venture capital firm Venrock. He also led the Series A and B rounds at Dollar Shave Club, which was acquired by Unilever for $1 billion. And, in 1991,
David co-created Apple Music when he was part of Apple’s system software product marketing group.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
</item 49>

<item 50>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 50>

<item 51>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Yahoo was already well behind Google in terms of search technology at that point. But also, Firefox was very well behind Chrome, which ended 2014 with nearly
50% market share and ended 2017 (when Google got prime default search position on Firefox) at nearly 65% of all web browser usage on desktop: and mobile is even
more skewed to Chrome. In other words, the number of Firefox users might have been declining for other reasons, although pushing the blame on Yahoo certainly
works in Google’s favor here!

The trial continues…
</item 51>

<item 52>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
The RSAs also motivate other companies to make or sell more Android devices, he said.

Expedia complains of too many ads on search, expensive ad payments

On November 1, Barry Diller, chair of Expedia and IAC, testified about his concerns regarding the increasing number of ads in search results having an impact on
organic listings.

“I must say I’m on the edge of revolt now that Google’s actions are so punitive, not just for Expedia but also for IAC and all the players that depend upon
something of a level playing field,” wrote Diller in a letter to Google back in 2019, according to Bloomberg.

Google CEO Sundar Pichai fired back that Google’s travel listings was one of the most popular experiences the company built.

Expedia execs also testified about the cost of ads and how increases had no impact on search results.
</item 52>

<item 53>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Google is now making it the default sign-in method for all users.

2 sec, let me text you some cash: When questioned about Mastercard’s prospects in emerging markets such as India, Mastercard’s CFO Sachin Mehra praised UPI for
helping with digitization but voiced reservations about its commercial sustainability, saying it is an “incredibly painful experience” for ecosystem
participants.

Bravely browsing — or searching — for a new job . . . : Brave Software, the maker of Brave Browser and Search, confirmed that it has laid off 9% of its
workforce across departments.
</item 53>

<item 54>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 54>

<item 55>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 55>

<item 56>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Battery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software that can be slipped into just about any
lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.

What lies beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched
on where startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.
</item 56>

<item 57>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Ray reassured me that I was not crazy. Google results today do feel different from how they felt just five or six years ago for two major reasons. The first was
Google’s response to the disinformation panic around the 2016 election, which involved questioning the notion that the most reliable information could be chosen
by a form of popularity, meaning how many links a site received from other sites. As a result, the algorithm seemed to change its approach to links, especially
when it came to news and sites offering legal, financial, or health advice, and instead paid more attention to what Google came to call E-E-A-T: experience,
expertise, authoritativeness, and trustworthiness.

“E-E-A-T has had a pretty big impact on what types of results you see,” Ray told me. She’s done extensive (and fascinating) research around how certain sites
have fared under these new guidelines: Urban Dictionary, down! Mayo Clinic, up! Some people consider EEAT part of what’s making results better than ever.
</item 57>

<item 58>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
This program, still in pilot testing, is open to all developers who sell apps in the 35 markets where it’s now available, including the U.S., and reduces the
standard commission by 4% for companies who choose to use their own payment processing solution. Spotify and Bumble were the initial testers for the new system,
first introduced in November 2022, and Match, as part of its settlement agreement, will also now take advantage of this option.

Google will also argue that its commissions aren’t just tied to payment processing, but rather help it to develop other features and controls to keep Android
users safe and guide them to discovering new applications, as well as help fund the tools and services developers need to build and grow their apps. Google also
helps developers reach a broader audience beyond the mobile phone, with support for Android across platforms, ranging from tablets to TVs to autos and more. And
it will note that 99% of its developers qualify for a commission of 15% or less, based on their revenues.
</item 58>

<item 59>
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
</item 59>

<item 60>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Looking at this from a less academic perspective, it’s worth noting that Google charges up to ten times more per transaction than you pay with PayPal or a
credit card, which does seem like a lot! And though Epic can’t argue it in court, I’ll point out that Apple wasn’t able to explicitly justify a 30 percent fee
to that case’s judge.

On the other hand, it does kinda seem like Epic wants something for nothing! As far as I can tell, Epic hasn’t stated what kind of fee would actually be
reasonable for Google to charge in exchange for placement in Google’s store — instead, Epic CEO Tim Sweeney seemingly suggests there shouldn’t be a Google fee
at all if a developer wants to use its own payment system.

How did Epic’s argument fare against Apple?

Well… both sides lost! But Epic arguably lost more.
</item 60>

<item 61>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Enter Google. The company revolutionized search by evaluating websites based on links from other websites, seeing each link as a vote of relevance and
trustworthiness. The founders pledged to be a neutral navigation system with no ads: just a clean white screen with a search box that would bring people off of
the Google landing page and out to a helpful website as seamlessly as possible. Users quickly decided this link-based sorting methodology was superior to the
existing search engines, and by the end of 1999, Google was handling the majority of online queries.

“I was basically just spamming Facebook with cars and articles about cars and sending traffic to banner ads, and that turned into $120,000 a month.”

Mapstead, like many of the early practitioners of SEO, figured out how to adapt. Almost as soon as Google took over, a secondary market emerged for links. For a
few hundred bucks, a firm in India or the Philippines could provide thousands of links from blog networks built entirely for that purpose.
</item 61>

<item 62>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
</item 62>

<item 63>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Tropic

Eliminated 26 roles, CEO David Campbell confirmed in a LinkedIn post October 21.

Convoy

Is halting operations due to a “massive freight recession,” CEO Dan Lewis announced in a memo October 19, a week after laying off 20% of its workforce.

LegalZoom

Plans to lay off more than 100 workers in its sales division, the company disclosed in a Worker Adjustment and Retraining Notification Act letter.

Plume

Laid off more than two dozen workers, TechCrunch learned October 18, about a year after the trans healthcare startup raised a $24 million Series B.

Google

Eliminated at least 40 roles in its news division, Google confirmed October 18.

Expedia Group

Cut around 100 jobs, Skift reports, in its second round of layoffs in recent months. The cuts reportedly impacted employees across data and AI.

Stack Overflow

Laid off 28% of its staff, more than likely impacting 100 people, the Prosus-owned company announced October 16.
</item 63>

<item 64>
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
</item 64>

<item 65>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
He gave Alan Eustace, Google’s senior vice president of engineering, what amounted to a blank check to hire any AI expertise he needed. Eustace and Jeff Dean,
who led the Brain lab, flew to Lake Tahoe and took Hinton and his students out to dinner at a steakhouse inside the hotel the night before the auction. The
smell of old cigarettes was overpowering, Dean recalled. They made the case for coming to work at Google. The next day, Hinton ran the auction from his hotel
room. Because of an old back injury, he rarely sat down. He turned a trash can upside down on a table, put his laptop on top and watched the bids roll in over
the next two days.

Google made an offer. So did Microsoft. DeepMind quickly bowed out as the price went up. The industry giants pushed the bids to $US20 million and then $US25
million, according to documents detailing the auction.
</item 65>

<item 66>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
she asked. “Or is this Google behaving differently than it used to?”

Sullivan had tried to convince me that Google was not behaving differently and, in fact, had not changed its search criteria in any major way for the past 20
years. Google wanted you to make good websites, and that was that. Everyone who tried to rank higher by messing with the algorithm would be blocked. Sullivan
even insisted that what these rule-breakers did should not be called SEO: he deemed it all “spam.” What is spam? “Spam is stuff that search engines don’t like.”

But the line between strategies that violate Google’s terms of service and strategies that don’t has always been blurry and inconsistently enforced. “I’ve never
seen this much tension in the industry in terms of, like, what Google says to do and what people are doing and getting away with,” Ray told me. “If you’re gonna
tell us that this stuff doesn’t work, make it stop working!”

Ray seemed like the most reasonable person I had spoken to so far.
</item 66>

<item 67>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Then, she offered to take Forrester for a joyride around Seattle. Obviously, he said yes.

Once he represented Bing, Forrester more or less stopped drinking at conferences, as had long been the case for his counterpart at Google, an engineer named
Matt Cutts, who helped build and then ran the company’s web spam team before stepping back in 2014 and leaving in 2016.

Cutts was a celebrity among SEOs, constantly mobbed with questions and complaints. When we spoke on the phone, he told me that before he left, he determined
that he had sent about 50,000 emails to people outside of Google during his decade and a half at the company.

Several SEOs described trying to get Cutts to drink at conferences so he would “spill secrets,” as one put it, but what generally ended up happening was that
all the SEOs would get drunk instead. Meanwhile, Cutts would stay sober, jotting down the latest SEO methods on a small notepad, sitting quietly in the corner
at the bar.
</item 67>

<item 68>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 68>

<item 69>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
The links that pop up when they go looking for answers online, they say, are “absolutely unusable”; “garbage”; and “a nightmare” because “a lot of the content
doesn’t feel authentic.” Some blame Google itself, asserting that an all-powerful, all-seeing, trillion-dollar corporation with a 90 percent market share for
online search is corrupting our access to the truth. But others blame the people I wanted to see in Florida, the ones who engage in the mysterious art of search
engine optimization, or SEO.

Doing SEO is less straightforward than buying the advertising space labeled “Sponsored” above organic search results; it’s more like the Wizard of Oz projecting
his voice to magnify his authority. The goal is to tell the algorithm whatever it needs to hear for a site to appear as high up as possible in search results,
leveraging Google’s supposed objectivity to lure people in and then, usually, show them some kind of advertising. Voilà: a business model!
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
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 71>

<item 72>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 72>

<item 73>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Anyone who thought the internet was already saturated with SEO-oriented content should buckle up.

“All the assholes that are out there paying shitty link-building companies to build shitty articles,” he said, “now they can go and use the free version of
GPT.” Soon, he said, Google results would be even worse, dominated entirely by AI-generated crap designed to please the algorithms, produced and published at
volumes far beyond anything humans could create, far beyond anything we’d ever seen before.

“They’re not gonna be able to stop the onslaught of it,” he said. Then he laughed and laughed, thinking about how puny and irrelevant Google seemed in
comparison to the next generation of automated SEO. “You can’t stop it!”

Once I was safe at home, my alligator attack bluster having deflated into an irrepressible affection for clever scoundrels, mixed with fear about the future
promised by said scoundrels, I decided to seek a broader range of the people who do SEO for a living.
</item 73>

<item 74>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
I see Deft as creating a new kind of UX for purchasing,” he said.

Startups like Neeva — which was founded by former Google execs — tried to convince people to switch to paid and private search with additional features.
However, earlier this year, the startup shut down its consumer efforts and eventually got acquired by Snowflake.

Pienaar said that Deft has to give compelling reasons to users to switch to its service.

“Users’ natural behavior for search is to follow the path of least resistance, i.e. most people just pick up their phone and just use the default search engine.
For Deft to change that behavior, they will need to create a high-value proposition,” he said.

Amir Konigsberg, an investor and founder of e-commerce search startup Twiggle (which shut down), said his startup wanted to solve ChatGPT-type interactions in
the shopping world. He said that the main problem for startups in this space is the go-to-market strategy and fundraising.
</item 74>

<item 75>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
SBF allegedly tried to buy off Trump: The TC team’s been trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced entrepreneur
accused of orchestrating the collapse of cryptocurrency exchange FTX. But fascinating details about SBF’s political dealings are emerging from a book by Michael
Lewis, “Going Infinite,” that debuted on the first day of the trial, like SBF’s attempt to buy off Trump to get him to not run again for president.

Gmail fights back against spammers: Google this week announced a series of significant changes to how it handles email from bulk senders in an effort to cut
down on spam and other unwanted emails. The company says that, starting next year, bulk senders will need to authenticate their emails, offer an easy way to
unsubscribe and stay under a reported spam threshold.

TikTok tests an ad-free tier: TikTok is testing an ad-free subscription tier for some users.
</item 75>

<item 76>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 76>

<item 77>
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
</item 77>

<item 78>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
In the beginning, the internet was seen as an improvement on the spirit of the public library. Here was an opportunity to transcend the gatekeepers controlling
who could publish a book, allowing mankind to fully connect and share knowledge. Instead, we have ended up in a situation arguably worse than before, where
nearly all online information runs through a single company, which assumes a veneer of civic utility, of impassive authority, when it is very much not a neutral
entity.

“There were so many true believers at Google in the early days,” Cutts told me. “As companies get big, it gets harder to get things done. Inevitably, people
start to think about profit or quarterly numbers.” He claimed that, at least while he was there, search quality always came before financial goals, but he
believes that the public underestimates how Google is shaping what they see, saying, “I deeply, deeply, deeply believe search engines are newspaper-like
entities, making editorial decisions.”
</item 78>

<item 79>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Alphabet

Google’s parent company announced laying off 6% of its global workforce on January 21, equating to 12,000 employees. These cuts impact divisions such as Area
120, the Google in-house incubator and Alphabet’s robotics division, Intrinsic.

Fandom

The entertainment company announced an unspecified number of employees impacted across multiple properties on January 20. According to a report by Variety, the
company employs around 500 people, and the layoffs have affected roughly 10% of its staff across different sites.

Swiggy

Announced plans to lay off 380 jobs on January 20 and shut down its meat marketplace.

Sophos

Announced on January 18, 10% of its global workforce, about 450 people were let go.

Microsoft

As announced on January 18, 10,000 employees will be impacted.

GoMechanic

Laid off 70% of its workforce on January 18.

Clearco

Announced on January 17, impacting 30% of staff across all teams.
</item 79>

<item 80>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Google’s software is constantly crawling the web and updating its central index, which itself is more than 100 million gigabytes in size. When you enter a
query, Search uses a few quick processes to decide the meaning of your question, including what kind of information you might be after. Then, beyond just
looking for sites with the words you typed in, the system uses machine learning to rank results according to what it thinks you’re looking for, prioritising
high quality and easy to use sites. If you allow it, Google does bring in some information it knows about you to further refine results, including your location
and web history. If you’re looking to turn this off, it’s in your Google account settings called “Web & App Activity”. Google says it does not infer personal
information like race or religion to shape results. Loading Google works with independent “search quality raters” to gauge the effectiveness of its processes,
and says it ran almost 900,000 quality tests in 2022.
</item 80>

<item 81>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
“It’s because they don’t have the real expertise.” Clay is an avuncular man with a mustache who is often credited with coining the phrase “search engine
optimization” and is therefore called “the father of SEO.” He told me his agency never hires an SEO with less than a decade of experience.

“I don’t know if you can trust anything you read online.”

Though Google publishes guidelines explaining how to do better in search (“Make your site interesting and useful”), the exact formula for how and why one
website gets placed over another is top secret, meaning that SEO involves a lot of reverse engineering and guesswork. With no clear chain of cause and effect
around why a site’s ranking has changed, a less talented practitioner can take on the mien of a premodern farmer, struggling to figure out how to make it rain.
Should he do that dance he did last year the night before it poured? Or maybe sacrifice his firstborn?
</item 81>

<item 82>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
“My favorite question to ask an SEO,” Cutts told me, was, “What’s the worst thing you’ve ever done?” which prompted responses that felt like “a cross between
showing off and a confessional.” So many SEOs were tempted to reveal the vulnerabilities they’d discovered in Google’s algorithms, even when they were talking
to the one person they really shouldn’t have been talking to, the guy who was planning to go back to his office and make those vulnerabilities disappear.

As a former SEO himself, Forrester understood that the quality of Bing’s search results would be impacted by the work of SEOs, so it made sense to communicate
with SEOs as much as possible. Cutts similarly tried to serve as a conduit between SEOs and Google, but Forrester felt that Google projected an attitude he
described as: “We know what we’re doing, we will stop your attempts to game this, and you know what? We’ll just kind of ignore you, and when you give us
feedback, eh, we don’t really care.”
</item 82>

<item 83>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.
</item 83>

<item 84>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The process of this test did remind me how much data Google collects — not just my searches but anything I say to a smart speaker, any search result or ad that
I click — which will be a point in favour of DuckDuckGo for some, even if it didn’t perform as well and is harder to read. The data collection does also mean
that Google (not just search but also mobile widgets and other services) personalises to me more effectively. Loading Bing has overhauled its layout over the
past year to be very graphic and augmented by generative AI explanations from its chatbot. I like the idea of being able to ask follow-up questions in searches
this way, but in practice it’s not really there yet. In a search for “kids BMX bike green” (Google did pretty well and DuckDuckGo ignored the green part), Bing
Chat offered a comparison of a few different options but they were too expensive. I asked, “any under $200?” and it replied with more options that were all over
$300.
</item 84>

<item 85>
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
</item 85>

<item 86>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
It was easy: buy links that led to your site and watch your ranking in Google’s results rise.

I came to understand that, since the dawn of the internet, there have been people attempting to manipulate search and then people decrying those manipulations
as the end of search’s ability to be useful. It works in cycles. People doing SEO find loopholes in the algorithm; critics complain about search results; search
engines innovate and close the loopholes. Rinse, repeat.

Before our current moment of widespread disillusionment with online information, the rise of SEO had reached a breaking point multiple times. In 2003, as Google
approached the deadline to disclose pertinent business information leading up to its IPO, the company quietly released an update cracking down. By 2011, SEO was
once again oppressively pervasive. TechCrunch published a story called “Why We Desperately Need a New (and Better) Google,” which argued that “Google has become
a jungle: a tropical paradise for spammers and marketers.”
</item 86>

<item 87>
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
</item 87>

<item 88>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
When Google bought DeepMind, it bought the whole thing. Musk was out. Financially, he had come out ahead, but he was unhappy. Three Google executives now firmly
in control of DeepMind were there: Page; Sergey Brin, a Google co-founder and Tesla investor; and Eric Schmidt, Google’s chair. Among the other attendees were
Reid Hoffman, another PayPal founder; and Toby Ord, an Australian philosopher studying “existential risk”.

The DeepMind founders reported that they were pushing ahead with their work but that they were aware the technology carried serious risks. Suleyman, the
DeepMind co-founder, gave a presentation called “The Pitchforkers Are Coming.” AI could lead to an explosion in disinformation, he told the board. He fretted
that as the technology replaced countless jobs in the coming years, the public would accuse Google of stealing their livelihoods. Google would need to share its
wealth with the millions who could no longer find work and provide a “universal basic income,” he argued. Musk agreed.
</item 88>

<item 89>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Quick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be the default search engine on Firefox. At
the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.

“I felt strongly that Yahoo was not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said
Yahoo had promised to reduce the number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.

“The number of users who stayed with Firefox declined noticeably during the years when Yahoo was the default,” said Baker.

Baker, who was providing a recorded deposition for Google’s defense, noted that Mozilla’s users apparently wanted and expected Google.

However, that is not the full story. Yahoo was already well behind Google in terms of search technology at that point.
</item 89>

<item 90>
The restaurant nearest Google (The Verge, 2023-10-26T13:00:00+00:00)
Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence. To improve local
ranking, Google encourages businesses to provide a trove of information on their profile. Owners should have basics like updated hours of operation and accurate
location details, but they should also respond to reviews, add photos, and even post individual items in stock to their Google page.

“I don’t think they’re really great marketing campaigns”

The prominence measure in Maps includes data from the web — links to the business, how the business ranks in traditional search, and press coverage, for
example. In this way, SEO impacts how a business appears in local search results, and news articles and links are one way Google tries to determine the
relevance of a page. (It’s why reporters’ inboxes are overrun with spammy requests to buy “guest posts” on our sites from marketers.)
</item 90>

<item 91>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
The search-engine giant has outperformed the broader market index with a 55% growth rate this year, surpassing the S&P 500's 12%.

Following the tech behemoth's upbeat Q2 results, investment firm Bank of America said it is expecting "favorable" headlines, due in part to strength in YouTube,
search, and artificial intelligence.

Just a few days before the earnings report, Seeking Alpha's Quant Rating system changed its view from Strong Buy to Hold, while Wall Street analysts maintained
a bullish view on the stock, giving it a Strong Buy rating.

Consensus EPS Estimates: $1.45

Consensus Revenue Estimates: $75.73B

Earnings Insight: Google has topped EPS expectations in 4 of the past 8 quarters, beating revenue estimates in 5 of those reports.

Microsoft (MSFT)

Joining Alphabet on Tuesday, Microsoft (MSFT) is also set to report its Q4 performance after the closing bell.
</item 91>

<item 92>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 92>

<item 93>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
In 2009, Sullivan was described as “the closest approximation to an umpire in the search world,” so when he published “A deep look at Google’s biggest-ever
search quality crisis” in 2017 and then took a job as Google’s public liaison for Search only a few months later, it felt to some SEOs as though a
congressperson working on gun safety legislation had quit to become an NRA lobbyist.

“There is a thread across the industry of people who believe that Google just made Danny an offer he couldn’t say no to, and it was designed essentially to take
his voice out of the conversation,” Forrester told me. “I don’t believe that’s the case,” he went on, but compared to Cutts, “I think that Danny specifically
stays out of a lot of public conversations because he is in those private conversations with businesses.”

Was all that really Google’s fault? Or the SEOs? Or was this about something deeper and more human: the will to exploit something so much we destroy it.
</item 93>

<item 94>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
The EU’s voluntary anti-disinformation Code has 44 signatories in all — covering not just major social media and search platforms such as the aforementioned
giants but entities from across the ad industry and civil society organizations involved in fact-checking.

Google

On generative AI, Google’s report discusses “recent progress in large-scale AI models” which it suggests has “sparked additional discussion about the social
impacts of AI and raised concerns on topics such as misinformation”. The tech giant is an early adopter of generative AI in search — via its Bard chatbot.

“Google is committed to developing technology responsibly and has published AI Principles to guide our work, including application areas we will not pursue,” it
writes in summary on the topic, adding: “We have also established a governance team to put them into action by conducting ethical reviews of new systems,
avoiding bias and incorporating privacy, security and safety.

“Google Search has published guidance on AI-generated content, outlining its approach to maintaining a high standard of information quality and the overall
helpfulness of content on Search.
</item 94>

<item 95>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
For $4.99, subscribers get an ad-free experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that
it’s piloting the plan in a single, English-speaking market outside the U.S. for now.

LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it throughout; an AI-powered LinkedIn Learning coach; and a
new AI-powered tool for marketing campaigns.

Musk comes clean about X’s metrics — maybe: In September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million
posts every day, excluding retweets. But speaking at an event this week, X CEO Linda Yaccarino offered a contradictory figure.
</item 95>

<item 96>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
“Maybe you ask a friend.”

After hearing so much about what it was like to be an SEO, I decided it was time to better understand what’s been going on from the perspective of the search
engine. Google was slow to allow someone to talk with me, possibly because of the giant PR clusterfuck that has been the company’s past year (accused by the
federal government of being a monopoly; increasingly despised by the public; losing ground to Reddit, TikTok, and large language models), so I decided to start
by meeting up with a chipper, charismatic man named Duane Forrester.

Forrester was at Microsoft from 2007 until 2015, where he helped launch and manage Bing, the perpetual underdog to Google’s domination of online search. Before
and after his time at Microsoft, Forrester worked as an SEO, so he sees the industry from both sides, like an aerospace engineer who spent a few years at the
Department of Defense, left for the private sector, and now is much better at winning military contracts.
</item 96>

<item 97>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 97>

<item 98>
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
</item 98>

<item 99>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Because they haven’t, he insisted. If anything, search results have gotten a lot better over time. Anyone who thought search quality was worse needed to take a
hard look in the mirror.

“We have an entire generation that grew up expecting the search box to do the work for them,” he said. “We might do a better job of matching for a bulk of
people, but for people who are super sensitive, when they have that fail moment, now it becomes, ‘All my searches aren’t good.’”

The problem was not Google. The problem was not SEOs. The problem was kids these days.

Of course Sullivan would say this, though. He works for Google. I felt like I began to understand why many SEOs had told me that Cutts’ departure had marked a
major turning point in the history of the internet, emblematic of Google’s transition from idealistic startup to one of the most valuable and powerful companies
to ever exist.
</item 99>

<item 100>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 100>
=== END ITEM id=4122a721d322f321 ===

=== ITEM id=9e3e810c1517d78e ===
Gold sentences and facts:
Sentence 0: When he entered free agency, a dozen teams lined up, curious to see if they could meet his eye.
  Fact 0.0: A player entered free agency.
  Fact 0.1: A dozen teams lined up when that player entered free agency.
  Fact 0.2: The teams were curious to see if they could meet that player's eye.
Sentence 1: As the player, I explore a system of the Imperium with a loyal crew of companions, making impactful choices and determining the fate of those around me.
  Fact 1.0: The speaker is the player.
  Fact 1.1: The player explores a system of the Imperium.
  Fact 1.2: The player has a loyal crew of companions.
  Fact 1.3: The player makes impactful choices.
  Fact 1.4: The player determines the fate of those around the player.

Retrieved items:
<item 1>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Old Trafford (and the training ground)

To find an apt shorthand for the lurching ship that is United, end the search at the roof of their stadium where on Manchester’s (many) wet days rain leaks
through. Old Trafford remains a storied venue but as it continues to crumble the tale of disrepair has become a central narrative in its recent history. As
Premier League titles have become scarce (zero in a decade) the tale of neglect of a ground that dates to 1910 is emblematic of the Glazers’ quasi‑absentee
proprietorship. Ratcliffe will invest $300m (£237m) into infrastructure but this is small beer when considering the multibillion-pound facelift or new venue
required. There is also the cramped Carrington base: as on Sir Matt Busby Way, the footprint is large at the training ground but the configuration and
facilities are ageing – badly.
</item 1>

<item 2>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Manchester United’s century old stadium needs an upgrade. Photograph: Greig Cowie/Shutterstock

Communications

The Glazers treat a microphone or TV camera as United fans do Liverpool or Manchester City: as a fierce enemy. No family member has given a press conference or
interview during their 18-year tenure.
</item 2>

<item 3>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
After all that pressure, all that tender hope, it looked a bit like a training goal.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

And that was pretty much that. From here United’s season continues to narrow. They can’t win the league now, can’t win the Champions League, can’t win the
Carabao Cup.
</item 3>

<item 4>
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
</item 4>

<item 5>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
Finally you get to manage the everyday metrics on which you will be judged; team, results, style, energy, messaging. Win things, improve players, create a
coherent and happy internal culture: but do it during an unceasing takeover circus while star players sulk and brief against you, while the ghost of Hamlet’s
father still skulks in the eaves. This felt like the end of one thing, perhaps even the beginning of the end of something else.
</item 5>

<item 6>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
All three players left Nice this summer. Newcastle’s sporting director, Dan Ashworth, and Paul Mitchell, Monaco’s sporting director until March, are among those
mentioned in dispatches as a potential Murtough replacement.

Can any CEO make United a best-in-class elite club again?

Blanc, who has an MBA from Harvard, is billed as the “Lionel Messi of business” by Nice’s chief executive, Fabrice Bocquet. At 60 years old Blanc has been the
Juventus CEO and president, CEO at Paris Saint‑Germain and French Tennis, and the executive overseeing the Tour de France. But once he or whoever becomes
Arnold’s successor is in place, it will dawn rapidly that United are a one-off hydra-headed beast of a club which is treacherously difficult to control. And
there is this, too: across town Manchester City have accrued six championships since United’s last in 2013 and boast a commercial operation that leaves
Ratcliffe’s new concern light years behind.
</item 6>

<item 7>
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
</item 7>

<item 8>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
In the second half United’s back five was André Onana, Aaron Wan-Bissaka, Jonny Evans, Raphaël Varane and Diogo Dalot, an entirely random collection of players
signed, at various times, by Ten Hag, José Mourinho, Ole Gunnar Solskjær and Alex Ferguson. Presumably Daley Thompson, Michael Portillo and Howard from Take
That were busy.

By the end United had five academy players on the pitch, ages ranged between 17 and 35, in outline evidence of internal good health, in reality another symptom
of a machine that wants to function properly but has for so long been made to judder along at half-speed. Ever feel like you’re starting to slip into the past
tense for good? When VAR happens at Old Trafford there isn’t even a VAR screen to say “VAR Check”. The 1990s were a hell of a decade. They still are around
here.
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=9e3e810c1517d78e ===

=== ITEM id=3110442c11b2523e ===
Gold sentences and facts:
Sentence 0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 0.0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 0.1: Paradigm was told that Bankman-Fried did not want investors as directors.
  Fact 0.2: Bankman-Fried planned on having a board with experts.
Sentence 1: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 1.0: Rehn declared to the jury that all of that, all of it, was built on lies.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud.
  Fact 2.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth.
  Fact 2.2: The prosecution said Bankman-Fried committed fraud to achieve power and influence.
  Fact 2.3: The defense argued that the FTX founder acted in good faith.
  Fact 2.4: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.5: The defense argued that the FTX founder basically got in over his head.

Retrieved items:

=== END ITEM id=3110442c11b2523e ===

=== ITEM id=9e5b9b00ed75d0ef ===
Gold sentences and facts:
Sentence 0: The post came after Swift was seen enthusiastically cheering him on in the box seats at Arrowhead Stadium, fuelling speculation that she and the athlete are dating.
  Fact 0.0: A post came after Swift was seen enthusiastically cheering on the athlete in the box seats at Arrowhead Stadium.
  Fact 0.1: The sighting of Swift cheering at Arrowhead Stadium fuelled speculation that Swift and the athlete are dating.
Sentence 1: Swifties were immediately on high alert, with some capturing Swift and Kelce leaving the stadium together.
  Fact 1.0: Swifties were immediately on high alert.
  Fact 1.1: Some Swifties captured Swift and Kelce leaving the stadium together.

Retrieved items:

=== END ITEM id=9e5b9b00ed75d0ef ===
