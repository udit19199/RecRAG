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

=== ITEM id=5441b1488f9ef97c ===
Gold sentences and facts:
Sentence 0: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 0.0: The parties referred to as 'they' get many other benefits from Android.
  Fact 0.1: Google could solve this problem today if Google wanted to.
Sentence 1: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 1.0: The case was filed by Helena World Chronicle.
  Fact 1.1: Helena World Chronicle is an Arkansas-based publisher.
  Fact 1.2: The case argues that Google siphons off news publishers' content.
  Fact 1.3: The case argues that Google siphons off news publishers' readers.
  Fact 1.4: The case argues that Google siphons off news publishers' ad revenue.
  Fact 1.5: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- report on Thursday -PUBLISHED_BY-> The Wall Street Journal
- Publishers -PARTICIPATED_IN-> class action antitrust suit
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal

Entities: class action antitrust suit, The Wall Street Journal, deal to license news for AI model training, News publisher files class action antitrust suit
against Google, citing AI’s harms to their bottom line, antitrust, The Atlantic, Axel Springer, TechCrunch, AI, Similarweb, Google, OpenAI
</item 1>

<item 2>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
That sounds fascinating, but if I don’t care about Fortnite or rich companies suing each other, what’s in it for me?

The future of Google’s app store could depend on this trial — both Epic and Google agree on that. Epic wants to break up Google’s alleged monopoly on Android
app stores and payment methods, so developers aren’t stuck paying the “Google Tax” or passing that fee along to you.

But if Epic wins — according to Google, anyhow — it could make Android phones less safe by dismantling basic protections against sideloaded apps, and damage
Android’s ability to compete with the iPhone because it (arguably) can’t run a competitive app store by giving it away for free.

Or maybe you just want juicy dirt on both companies? More on that later!

What exactly is Epic claiming?

Epic argues that Google makes it so difficult for developers and users to get around its Android app store, and that app store’s standard fees, that it’s
created an illegal monopoly that unjustly enriches Google while artificially driving up app prices because other stores can’t compete.

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android

Entities: Android, monopoly, Epic, Google, trial, Android app store, Google tax, Epic v. Google, explained, iPhone, The Verge
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Alphabet
- Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more -MENTIONS-> Alphabet
- ASX set to rise ahead of inflation report after big tech lifts Wall Street -MENTIONS-> Alphabet
- Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation -MENTIONS-> Alphabet
- Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas -MENTIONS-> Alphabet

Entities: Alphabet, Helena World Chronicle, U.S. District Court in D.C., Search Generative Experience, Bard AI chatbot, Knowledge Graph, anticompetitive
behavior, U.S. antitrust law, Sherman Act, Arkansas, question-and-answer technologies, News publisher files class action antitrust suit against Google, citing
AI’s harms to their bottom line, class action lawsuit, TechCrunch, AI technologies
</item 3>

<item 4>
Graph query row:
organization: Google
relationship: MENTIONS
article: The Epic v. Google trial may come down to simple v. complicated
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Helena World Chronicle
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. District Court in D.C.
- Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
-MENTIONS-> Washington Post
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Search Generative Experience

Entities: SGE, Google Bard AI, GoogleBot, Google general search service, AdSense, News Media Alliance, Washington Post, Epic Games lawsuit against Google over
app store antitrust issues, app store antitrust issues, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line,
class action antitrust suit against Google, Epic Games, Allen Institute for AI, TechCrunch, Google
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 6>

<item 7>
The best apps and tools for managing your money online (The Verge, 2023-11-12T13:00:00+00:00)
Hi, friends! Welcome to Installer No. 14, your guide to the best and Verge-iest stuff in the world. (If you’re new here, welcome, you’re my favorite, so happy
you’re here, and also, you can read all the old editions at the Installer homepage.)

I also have for you a rundown of the best tools for managing money, new gadgets from Valve and Humane, way too many writing apps, new stuff from ChatGPT, and
much more.

As always, the best part of Installer is your ideas and tips. What are you into right now? What have you been reading / watching / learning / doing that
everyone should know about? Tell me everything: installer@theverge.com. And if you know someone else who might enjoy Installer, forward it to them and tell them
to subscribe here.

Big week in the Installerverse! Let’s get to it.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, Valve, Installer, Humane, managing money, writing apps, The Verge, The best apps and tools for managing your money online
</item 7>

<item 8>
The best apps and tools for managing your money online (The Verge, 2023-11-12T13:00:00+00:00)
Big week in the Installerverse! Let’s get to it.

Installer / A weekly newsletter by David Pierce designed to tell you everything you need to download, watch, read, listen to, and explore that fits in The
Verge’s universe. Email (required) Sign up By submitting your email, you agree to our Terms and Privacy Notice . This site is protected by reCAPTCHA and the
Google Privacy Policy and Terms of Service apply.

The Drop

Group project

Last week, I asked you to share what systems you use for managing money. With the news that Mint was shutting down (which, ugh), I think a lot of people were
left suddenly looking for a new way to easily keep track of their budget and spending. I figured y’all might have some good ideas. (The Verge’s Barbara Krasnoff
also put together a list of Mint alternatives, and it’s a really good place to start.)

Once again, you came through!

Graph facts:
- Google -MENTIONS-> AI Principles
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Google
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Google
- Google -MENTIONS-> Apple

Entities: Google, managing money, The Verge, David Pierce, Installer, Barbara Krasnoff, Mint, reCAPTCHA, budget and spending, The best apps and tools for
managing your money online
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

Graph facts:
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal
- Europe’s artists and labels could lose $137m annually to US recorded music rightsholders if EU law isn’t changed, trade groups warn -MENTIONS-> Publishers
- A comprehensive list of 2023 tech layoffs -MENTIONS-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet

Entities: Knowledge Graph, Publishers, Knowledge Panel, Featured Snippets, class action antitrust suit, misappropriation of Publishers’ content, The Wall Street
Journal, report on Thursday, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line, antitrust, TechCrunch, AI,
Google
</item 9>

<item 10>
X plans to use payment, phone and ID verification to stop bots, in addition to the new $1/yr fee (TechCrunch, 2023-10-18T17:33:48+00:00)
A report this week from market intelligence firm Similarweb, for example, found that X’s web traffic and monthly active users on mobile had declined
post-acquisition.

Users today have far more options for microblogging and public discourse, thanks to products from Twitter-like startups and tech companies such as Spill,
Bluesky, Pebble, Countersocial, Spoutible, Hive and the open source platform Mastodon, as well as efforts from tech giants, like Meta’s new app Threads. Few of
the other social apps currently charge a fee, unless it’s for premium features, like some of the third-party Mastodon apps do.

In addition to presenting friction at sign-up, other critics have argued a fee could exclude genuine users and increase the digital divide.

Graph facts:
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> Threads
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Threads
- Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne -MENTIONS-> Threads
- 2023 in social media: the case for the fediverse -MENTIONS-> Threads
- Pebble, the Twitter alternative previously known as T2, is shutting down -MENTIONS-> Threads

Entities: Threads, Meta, TechCrunch, X, bots, X plans to use payment, phone and ID verification to stop bots, Similarweb, Countersocial, Spoutible, Hive,
microblogging, public discourse, $1/yr fee, digital divide, Spill
</item 10>
=== END ITEM id=5441b1488f9ef97c ===

=== ITEM id=4726e2170fd84c0d ===
Gold sentences and facts:
Sentence 0: James is seeking $250 million in penalties and a ban on Trump doing business in New York, his home state.
  Fact 0.0: James is seeking $250 million in penalties.
  Fact 0.1: James is seeking a ban on Trump doing business in New York.
  Fact 0.2: New York is Trump's home state.
Sentence 1: No apartment in New York City has ever sold for close to that amount, James said.
  Fact 1.0: No apartment in New York City has ever sold for close to that amount, according to James.
Sentence 2: The prosecution argues that was to mask a drop in the value of one of his other properties.
  Fact 2.0: The prosecution argues that a transaction was meant to mask a drop in the value of one of Trump's other properties.

Retrieved items:
<item 1>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
James, a Democrat, sued Trump and the Trump Organization a year ago, alleging a pattern of duplicity that she dubbed “the art of the steal,” a twist on the
title of Trump’s 1987 business memoir “The Art of the Deal.”

The lawsuit accused Trump and his company of routinely inflating the value of assets like skyscrapers, golf courses and his Mar-a-Lago estate in Florida,
padding his bottom line by billions.

Among the allegations were that Trump claimed his Trump Tower apartment in Manhattan — a three-story penthouse replete with gold-plated fixtures — was nearly
three times its actual size and valued the property at $327 million. No apartment in New York City has ever sold for close to that amount, James said.

Trump valued Mar-a-Lago as high as $739 million — more than 10 times a more reasonable estimate of its worth.

Graph facts:
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> New York City
- The people who ruined the internet -MENTIONS-> New York City
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -MENTIONS-> New York City
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> New York City
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> New York City

Entities: New York City, Fortune, lawsuit, Manhattan, fraud, Donald Trump, Trump Organization, The Art of the Deal, inflating asset values, Mar-a-Lago, Florida,
Trump Tower apartment, James, Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former
president
</item 1>

<item 2>
Attorney general going after Jeffrey Epstein's estate says she was fired for her dogged pursuit: 'My bar license, my integrity were more important to me'
(Fortune, 2023-11-17T21:59:14+00:00)
The place is empty. On the silver chain round her neck hangs a gift from her husband: a pendant in the shape of the scales of justice.

Epstein’s estate was the first to settle with the Virgin Islands. A year ago, it agreed to pay $105 million, plus half the proceeds from the sale of Little St.
James, AKA “Little St. Jeff,” the private island where authorities allege some of Epstein’s crimes took place.

Then, last January, Wall Street billionaire Leon Black settled for $62.5 million. Next came JPMorgan Chase., in September, for $75 million. None of the parties
admitted to wrongdoing. All of them declined to comment for this story.

Without those cases, the world may have never glimpsed some of the secrets Epstein and his circle harbored, or the depths of his influence here in the
Caribbean.

Graph facts:
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> JPMorgan Chase
- Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison -MENTIONS-> JPMorgan Chase
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-MENTIONS-> JPMorgan Chase
- Jamie Dimon -WORKS_FOR-> JPMorgan Chase
- JPMorgan Chase -ACCUSED_OF-> violating federal laws

Entities: JPMorgan Chase, Fortune, Caribbean, Jeffrey Epstein, Attorney general, Epstein’s estate, Little St. James, Leon Black, Epstein’s crimes, Epstein
estate settlement with the Virgin Islands, Leon Black settlement, JPMorgan Chase settlement, Attorney general going after Jeffrey Epstein's estate says she was
fired for her dogged pursuit: 'My bar license, my integrity were more important to me', Virgin Islands
</item 2>

<item 3>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
Engoron’s ruling, in a phase of the case known as summary judgment, resolves the key claim in James’ lawsuit, but six others remain.

Engoron is slated to hold a non-jury trial starting Oct. 2 before deciding on those claims and any punishments he may impose. James is seeking $250 million in
penalties and a ban on Trump doing business in New York, his home state. The trial could last into December, Engoron has said.

Trump’s lawyers had asked the judge to throw out the case, which he denied. They contend that James wasn’t legally allowed to file the lawsuit because there
isn’t any evidence that the public was harmed by Trump’s actions. They also argued that many of the allegations in the lawsuit were barred by the statute of
limitations.

Engoron, noting that he had “emphatically rejected” those arguments earlier in the case, equated them to the “time-loop in the film ‘Groundhog Day.'”

Graph facts:
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> New York
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> New York
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> New York
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> New York
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> New York

Entities: New York, Fortune, defrauding banks, Engoron, non-jury trial, Groundhog Day, statute of limitations, public harm, trial could last into December,
Donald Trump, banks, James, James’ lawsuit, Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation
against the former president
</item 3>

<item 4>
Attorney general going after Jeffrey Epstein's estate says she was fired for her dogged pursuit: 'My bar license, my integrity were more important to me'
(Fortune, 2023-11-17T21:59:14+00:00)
In the end the estate returned $80 million in tax incentives, agreed to pay half a million to remediate environmental damage to one of Epstein’s islands,
committed half the proceeds from the future sale of Little St James and established a victims’ compensation fund.

Officially, George, not the governor, was in charge of making the legal calls. But Bryan was still her boss and held the purse strings. It was a delicate
balance, George said.

Trying to untangle a half-billion-dollar estate is a formidable task under the best of circumstances. It was even harder given the code of silence still
surrounding the dead financier. Many of Epstein’s employees had signed nondisclosure agreements. Some alerted the estate’s lawyers the moment investigators
reached out. On Little St. Jeff, groundskeepers avoided eye contact when authorities approached.

George herself traveled to Little St. Jeff. She still can’t shake eerie absurdity of seeing a dentist’s chair inside the luxury island villa.

Graph facts:
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> the governor
- the governor -SAID-> Kentucky sports betting
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune

Entities: the governor, Fortune, Jeffrey Epstein's estate, Jeffrey Epstein, Little St. Jeff, Attorney general going after Jeffrey Epstein's estate says she was
fired for her dogged pursuit: 'My bar license, my integrity were more important to me', Bryan, George, Little St James, luxury island villa, dentist's chair,
tax incentives, environmental damage, victims' compensation fund, nondisclosure agreements
</item 4>

<item 5>
The $777 million surprise: Donald Trump is getting richer (The Age, 2023-11-07T22:22:05+00:00)
They haven’t provided a more recent estimate.

Space inflation wasn’t the only issue. Between 2014 and 2015, Trump increased the value of the apartment to $US327 million from $US200 million. The prosecution
argues that was to mask a drop in the value of one of his other properties. “A discrepancy of this order of magnitude, by a real estate developer sizing up his
own living space of decades, can only be considered fraud,” Judge Arthur Engoron wrote in a September ruling. ‘The company has never been stronger and never
been better.’ Eric Trump Bloomberg values the property using recent Trump Tower sales, as well as those of comparable Manhattan penthouses. While some splashy
residences in much newer buildings on so-called Billionaires Row have listed for well in excess of $US100 million, they typically boast higher ceilings, more
up-to-date finishes, contemporary layouts, outdoor space and better views.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, Manhattan, Arthur Engoron, fraud, Donald Trump, The $777 million surprise: Donald Trump is getting richer, Trump Tower, Eric Trump,
Billionaires Row, Bloomberg
</item 5>

<item 6>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
A judge ruled Tuesday that Donald Trump committed fraud for years while building the real estate empire that catapulted him to fame and the White House.

Judge Arthur Engoron, ruling in a civil lawsuit brought by New York’s attorney general, found that the former president and his company deceived banks, insurers
and others by massively overvaluing his assets and exaggerating his net worth on paperwork used in making deals and securing financing.

Engoron ordered that some of Trump’s business licenses be rescinded as punishment and said he would continue to have an independent monitor oversee the Trump
Organization’s operations.

A Trump spokesperson did not immediately respond to a request for comment on the ruling. Trump has long insisted he did nothing wrong.

The decision, days before the start of a non-jury trial in Attorney General Letitia James’ lawsuit, is the strongest repudiation yet of Trump’s carefully
coiffed image as a wealthy and shrewd real estate mogul turned political powerhouse.

Graph facts:
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> New York
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> New York
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> New York
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> New York
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> New York

Entities: New York, real estate, Fortune, White House, Arthur Engoron, Letitia James, fraud, civil lawsuit, non-jury trial, Donald Trump, Trump Organization,
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president
</item 6>

<item 7>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
The company was fined $1.6 million. One of the executives, Trump’s longtime finance chief Allen Weisselberg, pleaded guilty and served five months in jail. He
is a defendant in James’ lawsuit and gave sworn deposition testimony for the case in May.

James’ lawsuit does not carry the potential of prison time, but could complicate his ability to transact real estate deals. It could also stain his legacy as a
developer.

James has asked Engoron to ban Trump and his three eldest children from ever again running a company based New York. She also wants Trump and the Trump
Organization barred from entering into commercial real estate acquisitions for five years, among other sanctions. The $250 million in penalties she is seeking
is the estimated worth of benefits derived from the alleged fraud, she said.

Graph facts:
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> New York
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> New York
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> New York
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> New York
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> New York

Entities: New York, real estate, Fortune, fraud, Engoron, Donald Trump, Trump Organization, Allen Weisselberg, James, James’ lawsuit, Donald Trump defrauded
banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president
</item 7>

<item 8>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
James, who campaigned for office as a Trump critic and watchdog, started scrutinizing his business practices in March 2019 after his former personal lawyer
Michael Cohen testified to Congress that Trump exaggerated his wealth on financial statements provided to Deutsche Bank while trying to obtain financing to buy
the NFL’s Buffalo Bills.

James’ office previously sued Trump for misusing his own charitable foundation to further his political and business interests. Trump was ordered to pay $2
million to an array of charities as a fine and the charity, the Trump Foundation, was shut down.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Fortune, Congress, Buffalo Bills, defrauding banks, Donald Trump, James, Donald Trump defrauded banks with 'fantasy' to build his real estate
empire, judge rules in a major repudiation against the former president, Michael Cohen, Deutsche Bank, James' office, Trump Foundation, Trump's business
practices, misusing a charitable foundation, scrutinizing Trump's business practices
</item 8>

<item 9>
Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
Beyond mere bragging about his riches, Trump, his company and key executives repeatedly lied about them on his annual financial statements, reaping rewards such
as favorable loan terms and lower insurance premiums, Engoron found.

Those tactics crossed a line and violated the law, the judge said, rejecting Trump’s contention that a disclaimer on the financial statements absolved him of
any wrongdoing.

“In defendants’ world: rent regulated apartments are worth the same as unregulated apartments; restricted land is worth the same as unrestricted land;
restrictions can evaporate into thin air; a disclaimer by one party casting responsibility on another party exonerates the other party’s lies,” Engoron wrote in
his 35-page ruling. “That is a fantasy world, not the real world.”

Manhattan prosecutors had looked into bringing a criminal case over the same conduct but declined to do so, leaving James to sue Trump and seek penalties that
could disrupt his and his family’s ability to do business in the state.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Manhattan, Trump's company, key executives, Manhattan prosecutors, defrauding banks, lying on annual financial statements, disclaimer
absolving Trump of wrongdoing, tactics crossing a line and violating the law, criminal case over the same conduct, Trump's family, Engoron, Donald Trump, banks,
James
</item 9>

<item 10>
Graph query row:
article: Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president
publisher: Fortune
person: Letitia James
relationship: MENTIONS
related_entity: NFL
passage: Donald Trump defrauded banks with 'fantasy' to build his real estate empire, judge rules in a major repudiation against the former president (Fortune,
2023-09-26T21:11:15+00:00)
A judge ruled Tuesday that Donald Trump committed fraud for years while building the real estate empire that catapulted him to fame and the White House.

Judge Arthur Engoron, ruling in a civil lawsuit brought by New York’s attorney general, found that the former president and his company deceived banks, insurers
and others by massively overvaluing his assets and exaggerating his net worth on paperwork used in making deals and securing financing.

Engoron ordered that some of Trump’s business licenses be rescinded as punishment and said he would continue to have an independent monitor oversee the Trump
Organization’s operations.

A Trump spokesperson did not immediately respond to a request for comment on the ruling. Trump has long insisted he did nothing wrong.

The decision, days before the start of a non-jury trial in Attorney General Letitia James’ lawsuit, is the strongest repudiation yet of Trump’s carefully
coiffed image as a wealthy and shrewd real estate mogul turned political powerhouse.
</item 10>
=== END ITEM id=4726e2170fd84c0d ===

=== ITEM id=db7211ca89921c64 ===
Gold sentences and facts:
Sentence 0: The All Blacks had their chances, but their usual safe handling, a feature throughout the World Cup after their first-up loss to France, evaded them against the Springboks’ blitz defence in the wet conditions.
  Fact 0.0: The All Blacks had their chances.
  Fact 0.1: The All Blacks' usual safe handling was a feature throughout the World Cup.
  Fact 0.2: The All Blacks' usual safe handling evaded them against the Springboks' blitz defence in the wet conditions.
  Fact 0.3: The All Blacks lost their first-up World Cup match to France.
Sentence 1: It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.
  Fact 1.0: The situation described comes despite Argentina beating the All Blacks last year in Christchurch.
  Fact 1.1: The situation described comes despite a maiden defeat in 2020 in Sydney.

Retrieved items:
<item 1>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
A missed Richie Mo’unga conversion following Beauden Barrett’s 58th minute try – the only five-pointer of the match – ultimately proved the difference, as
Handre Pollard’s four three-pointers led the Springboks to World Cup glory.

The All Blacks had their chances, but their usual safe handling, a feature throughout the World Cup after their first-up loss to France, evaded them against the
Springboks’ blitz defence in the wet conditions.

All night the All Blacks’ timing was off and the forced ball from Savea out wide just as the men in black saw space and looked to open up the Springboks in the
80th minute proved to be killer blow.

Wayne Barnes, who was booed relentlessly, called a scrum reset as the seconds ticked by.

Advertisement

Moments later, their hearts sunk as the Springboks managed to win the scrum and then hang onto the ball.

Graph facts:
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Rassie Erasmus -WORKS_FOR-> Springboks
- Jacques Nienaber -WORKS_FOR-> Springboks
- Erasmus -WORKS_FOR-> Springboks

Entities: Springboks, Richie Mo’unga, Beauden Barrett, Handre Pollard, Savea, Wayne Barnes, World Cup final, THE GREATEST! Boks beat All Blacks in instant
classic to claim record FOURTH RWC in controversial, dramatic final, France, All Blacks, The Roar | Sports Writers Blog
</item 1>

<item 2>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
Advertisement

Robertson will have an almighty task on his hand, with multiple generational players including Dane Coles, Sam Whitelock, Brodie Retallick and Aaron Smith to
call time on their international careers. Others like Ardie Savea, Mo’unga, Beauden Barrett and Shannon Frizell will also head to Japan.

While Foster didn’t want to take anything away from the Springboks’ win, he said the game had some issues to sort out regarding the officiating.

“I’m proud of our guys, to go down to that red card so early and fight our way back and give ourselves a chance is pretty special,” the departing All Blacks
coach said.

“I don’t think it went wrong in any clear [place on the pitch], it was a real arm wrestle, both teams have their moment, incredibly proud of the way we fought
and to get within a whisker of pulling it off is heart-breaking.

Graph facts:
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Ardie Savea -PARTICIPATED_IN-> World Cup quarter-final
- Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues -MENTIONS-> Ardie Savea
- Rassie Erasmus -WORKS_FOR-> Springboks

Entities: Ardie Savea, Springboks, Beauden Barrett, THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic
final, Robertson, Sam Whitelock, Brodie Retallick, Shannon Frizell, officiating, RWC final, Mo’unga, Dane Coles, Foster, Japan, All Blacks
</item 2>

<item 3>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
With moments left in extra time against the South Americans, Arnold dragged Mat Ryan and replaced him with Andrew Redmayne – who saved the decisive spot kick
(and also threw his opponents water bottle with cheat sheet attached away from the goal).

Advertisement

Ryan might well have gotten the Socceroos to Qatar, but if Australia had not advanced there’s little chance Arnold would still be their coach today.

And there’s virtually no chance he’ll have to endure a similar level of stress through to the next finals – with FIFA expanding the tournament from 36 to 48
teams meaning eight direct Asian confederation qualifiers will proceed directly instead of four.

Arnold, in his second stint as national team manager, will prowl the technical area for a 59th time on Thursday, taking him clear of Frank Farina.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Australia
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Australia
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Australia
- Israel's defense undone by reliance on technology -MENTIONS-> Australia
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Australia

Entities: Australia, ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment, Socceroos, Arnold, Mat Ryan, Andrew
Redmayne, South Americans, the tournament, Asian confederation, Frank Farina, Qatar, FIFA, The Roar
</item 3>

<item 4>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
STADE DE FRANCE – Twenty-eight years between drinks and drama was always on the cards between these two great foes. Little did we know just how much.

But just like that famous win at Ellis Park where Nelson Mandela handed over the William Webb Ellis Cup to Francois Pienaar, it was the Springboks who once
again prevailed 12-11 in an instant classic.

Unlike the blue skies that greeted Johannesburg and the Boeing 747 plane that sent shivers down the spines of everyone in attendance in 1995, rain and dampness
poured down on the Stade de France.

Facebook Twitter Whatsapp Reddit Email Share

Yet, nothing was going to dampen this evening as 80,065 fans poured in.

In a match of controversy, where cards of yellow and red were handed out quicker than a game of Uno, thundering shots both legal and illegal were put on by
rugby’s two greatest sides.

Graph facts:
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> Springboks
- Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang -MENTIONS-> Springboks
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> William Webb Ellis Cup
- Subtle as a sledgehammer, a pop icon returns -MENTIONS-> Nelson Mandela
- Rassie Erasmus -WORKS_FOR-> Springboks

Entities: 1995 win at Ellis Park, Nelson Mandela, Francois Pienaar, William Webb Ellis Cup, Boeing 747, Springboks, THE GREATEST! Boks beat All Blacks in
instant classic to claim record FOURTH RWC in controversial, dramatic final, RWC final, Stade de France, Ellis Park, Johannesburg, All Blacks, The Roar
</item 4>

<item 5>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
The All Blacks are 18-point favourites for the opening semi-final.

It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.

“The past sets us up beautifully for both teams,” Foster said.

“We do know each other but we don’t know each other terms of a Rugby World Cup, so at this stage in the tournament it’s new territory for us and as we’ve seen
World Cups are very different.”

“We have learnt to greatly respect Argentina. They have a rich history of overachieving at the World Cup and have done a fantastic job to get here at the same
level we are, so it is going to be a heck of a game.”

Advertisement

Foster said the All Blacks, who suffered a shock semi-final loss to Eddie Jones’ England in 2019, never consider themselves favourites.

“Firstly, you’ve never heard us say we’re favourites,” he said.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Cane
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Whitelock
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Brodie Retallick
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Mark Telea
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Johnny Sexton

Entities: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand, Christchurch, Opening semi-final,
Argentina-All Blacks match last year, Argentina-All Blacks match, All Blacks-England semi-final, England, Foster, Sydney, Rugby World Cup, Eddie Jones, All
Blacks, Argentina, New Zealand, The Roar
</item 5>

<item 6>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
“This group is very proud to be All Blacks, very proud to play for their country and have worked incredibly hard to get here.

“While I am so proud of what we did and how we worked we have to give it to South Africa, they are a quality team. They have had a few close victories in this
tournament and are a different class. It’s special for them but it equally hurts for us. I can’t ask for much more than we did, we couldn’t put more out on the
park.”

Advertisement

Controversy was on display from the outset, as Frizell was shown a yellow card for an illegal cleanout on Springboks hooker Bongi Mbonambi. The incident saw the
hooker, who was cleared earlier in the week by World Rugby following an alleged racial slur on England back-rower Tom Curry, replaced by Deon Fourie in the
third minute.

The penalty allowed Pollard to step up and bang over three points.

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -PUBLISHED_BY-> The Roar
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> 1995 win at Ellis Park
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Nelson Mandela
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Francois Pienaar
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> William Webb Ellis Cup

Entities: THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final, Frizell, Bongi Mbonambi, Tom Curry,
Deon Fourie, illegal cleanout, racial slur, RWC final, South Africa, England, Pollard, World Rugby, All Blacks, The Roar | Sports Writers Blog
</item 6>

<item 7>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
In March last year Graham Arnold seemed a dead man walking – the Socceroos World Cup qualification campaign was lurching towards failure and the manager was
being battered from all sides.

Arnold’s team had finished seven points off automatic qualification for Qatar after a shocking run of one win in seven games left them facing a Asian playoff
against United Arab Emirates and then another against Peru to make the finals.

While rumours swirled and ex-players and fans bayed for his blood, Football Australia took the pragmatic course and stuck by Arnold. It likely had more to do
with the lateness of the hour and financial considerations than any overwhelming faith in Arnie to run around the fortunes.

Facebook Twitter Whatsapp Reddit Email Share

But he did just that with qualification – and an excellent run at the finals – bowing out to the eventual champions Argentina and after nearly taking them to
extra time.

Graph facts:
- Arnold -PARTICIPATED_IN-> playoff against Peru
- ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment -MENTIONS-> Australia
- ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment -PUBLISHED_BY-> The Roar | Sports Writers Blog
- Arnold -WORKS_FOR-> Socceroos
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Peru

Entities: ‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment, Graham Arnold, Socceroos, Socceroos World Cup
qualification campaign, Asian playoff against United Arab Emirates, playoff against Peru, Peru, World Cup, Football Australia, World Cup finals, Qatar,
Argentina, The Roar, United Arab Emirates
</item 7>

<item 8>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
Advertisement

“In the game in 2020 there were things you could pinpoint. You can’t say that one game means we can do it as there are a lot of games where we haven’t.

“That moment has to be just a part of it. What we have done since then, all the success and failures [matters in our preparation]. The other team are heavy
favourites, everyone is expecting a New Zealand and South Africa final. We just have to focus on what is required.”

Cheika has maintained that his Los Pumas side was too tight earlier in the tournament but has begun to free themselves up since qualifying for the knockout
stages – and rewarding their fans by doing so.

On inspiring the country?

“It’s not easy,” he said.

“We want to inspire not just by our results but by our ambition, by our ability to get back up and overcome obstacles.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Cane
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Whitelock
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Brodie Retallick
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Mark Telea
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Johnny Sexton

Entities: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand, Cheika, Los Pumas, South Africa, All
Blacks, New Zealand, The Roar
</item 8>

<item 9>
THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final (The Roar | Sports Writers Blog,
2023-10-28T21:06:03+00:00)
Minutes later and Kolisi was off, as he cannoned into Savea. He would later return, much to the anger of the New Zealanders in the crowd who were calling for
justice.

Sensing an opportunity to strike against 14 men, the All Blacks kicked for the corner on a couple occasions out wide. It came back to haunt them, as their
usually silky and composed skill sets failed them.

That much was true when the TMO intervened after Smith thought he had a try in his final Test, but an error at the lineout cancelled out the five-pointer.

Advertisement

Eventually the All Blacks did have a try, as Mark Telea, the All Blacks’ best and biggest attacking threat all night, managed to get on the outside and draw in
two defenders before offloading to Beauden Barrett. But Mo’unga missed the conversion to leave them trailing by a point.

Graph facts:
- Beauden Barrett -PARTICIPATED_IN-> World Cup final
- Beauden Barrett -PARTICIPATED_IN-> Test match
- Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues -MENTIONS-> Beauden Barrett
- Legacy vs ego: The Springboks are playing for South Africa, the All Blacks are playing for themselves -MENTIONS-> Beauden Barrett
- Savea -PARTICIPATED_IN-> World Cup final

Entities: Boks, Beauden Barrett, Savea, THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final, RWC
final, Mo’unga, Kolisi, Smith, Mark Telea, All Blacks, The Roar | Sports Writers Blog
</item 9>

<item 10>
‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand (The Roar | Sports Writers Blog,
2023-10-18T20:55:55+00:00)
“Firstly, you’ve never heard us say we’re favourites,” he said.

‘We know these games are do or die. We’re going to have to improve our performance. The stage gets bigger at this time in the tournament and you’ve got to grow
your game. That’s our strong focus. The best team on the night wins – Argentina has done that to us.”

Cheika however was happy to remind the All Blacks who were favourites.

“”New Zealand has always been an example in rugby, a benchmark. They make you think of high-level skills in a really open game but there are always threats in
the lineout, mauls and rucks,” said Cheika, whose Wallabies lost in the 2015 final.

“You have to take the moments you are in the game and see them coming at you at different times.

Advertisement

“In the game in 2020 there were things you could pinpoint.

Graph facts:
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Cane
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Sam Whitelock
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Brodie Retallick
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Mark Telea
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Johnny Sexton

Entities: ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand, Cheika, Argentina, 2015 final, game in
2020, Wallabies, rugby, All Blacks, New Zealand, The Roar
</item 10>
=== END ITEM id=db7211ca89921c64 ===
