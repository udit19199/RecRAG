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

=== ITEM id=7135fb5a974aa21c ===
Gold sentences and facts:
Sentence 0: While his Republican colleagues have largely shunned TikTok, Ramaswamy is presenting himself as one of the Cool Politicians who will actually use it.
  Fact 0.0: Ramaswamy's Republican colleagues have largely shunned TikTok.
  Fact 0.1: Ramaswamy is presenting himself as one of the Cool Politicians.
  Fact 0.2: The Cool Politicians are politicians who will actually use TikTok.
Sentence 1: Artists have spent the last few years trying to drum up engagement for their work by making their songs trend on TikTok, which audiences have started to resist.
  Fact 1.0: Artists have spent the last few years trying to drum up engagement for their work by making their songs trend on TikTok.
  Fact 1.1: Audiences have started to resist artists making their songs trend on TikTok.

Retrieved items:
<item 1>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
She claimed X was seeing 500 million posts per day on the platform. So who’s right? Beats us.

Former NSA director’s startup shutters: IronNet, a once-promising cybersecurity startup founded by a former NSA director, has shuttered and laid off its
remaining staff following its collapse. The Virginia-based IronNet was founded in 2014 by retired four-star general Keith Alexander and had raised more than
$400 million in funding. But IronNet failed to gain traction after going public in August 2021, and its stock price continued to steeply decline in the wake of
an initial spike.

Audio

On the hunt for a new podcast to listen to while you work out, do the dishes or rake the leaves (now that fall’s arrived)? Look no further than TechCrunch’s
roster, which covers the world of startups, the blockchain and more.
</item 1>

<item 2>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Ken Russell, a Democrat who left the Miami City Commission to run for a House seat in 2022, leaned into the cringe with bait-and-switch thirst traps reminding
viewers to vote. In another video titled “Appealing to the youth vote,” he recreated Steve Buscemi’s “How do you do, fellow kids?” to encourage voter
registration.

Audience engagement doesn’t rely on forcing fun. North Carolina Rep. Jeff Jackson updates constituents on TikTok, breaking down topics like the government
shutdown in concise explanatory videos without the frills of internet humor. Even though his content is less exciting, his account has over 2.2 million
followers.

As an alternative to the politician-influencer, some campaigns are focusing on mobilizing creators who already have an engaged audience. Biden is not on TikTok,
but his digital strategy team is building an “army of influencers” to reach viewers who wouldn’t typically keep up with the White House press corps.
</item 2>

<item 3>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
To produce TikTok’s new report, which you can read in full here, Luminate researchers studied data from the US, UK, Germany, Brazil and Indonesia, as well as
global data, for a 12-month period from July 1, 2022 to June 30, 2023, and found that “compared to average consumers, TikTok users are significantly more likely
to be music streamers”.

We know that TikTok counted 150 million users in the United States as of February 2023, which means that, if we look at this stat through the lens of the
Luminate analysis, then at this point it appears that approximately 93 million TikTok users in the US are paying for a subscription to one of the various
streaming services available in the United States, including Spotify, Apple Music and others.
</item 3>

<item 4>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Battery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software that can be slipped into just about any
lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.

What lies beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched
on where startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.
</item 4>

<item 5>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Critics on both the right and the left described the launch as a disaster.

Conservative organizations have proven especially out of their depth when facing the wrath of extremely online social media users, who mobilize trolling for
social justice. In 2020, TikTok users reserved hundreds of tickets for a Trump rally and never showed up, leaving the president to face swaths of empty seats.
When a Texas anti-abortion group opened an anonymous tip form to enforce the state’s ban on abortion after six weeks, Gen Z activists flooded the site with
Shrek porn. This year, TikTok and Twitter users shut down the Missouri attorney general’s tip form for reporting clinics that provide gender-affirming care.
Within days of launching, the site was spammed with fanfiction, satirical anecdotes about kids getting “transed” and fanfiction. The attorney general’s press
secretary blamed the site’s downfall on activists “hacking the system.”
</item 5>

<item 6>
VTubers have reached their inevitable conclusion (Polygon, 2023-10-19T16:01:50+00:00)
“Let people fail and still have that opportunity for growth in the future. More risk-taking and trying to branch out in other ways instead of simply just
streaming or following preestablished methods.”

According to both of the people we interviewed for this story, this is an agreed-upon sentiment — that to prevent further stagnation, this not-so-niche form of
content creation needs to reinvent itself and allow more freedom to its talents. Time will tell if larger agencies like Hololive (which reported 50% growth in
its 2023 financial report) and Nijisanji will allow their talents to break the mold.
</item 6>

<item 7>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Startups, he added, can use speed to their advantage as they work to “innovate, iterate and deploy solutions” to customers.

Will that work prove defensible in the long run? Edward Tsai, a managing partner at Alumni Ventures, told us that he had a potentially “controversial opinion
that VCs and startups may want to temporarily reduce their focus on defensibility and increase their focus on products that deliver compelling value and
focusing on speed to market.” Presuming massive TAM, that could work!

Read on for answers to all our questions from:

Rick Grinnell, founder and managing partner, Glasswing Ventures

There are several layers to the emerging LLM stack, including models, pre-training solutions and fine-tuning tools. Do you expect startups to build striated
solutions for individual layers of the LLM stack, or pursue a more vertical approach?
</item 7>

<item 8>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
“Luminate’s findings prove what we had long known: that TikTok is the driving force behind music discovery in the industry, and that TikTok users are active,
engaged and highly valuable drivers of music industry revenues.” Ole Obermann, TikTok

This past summer, TikTok launched its own music streaming service, TikTok Music, a premium-only service initially available in Brazil and Indonesia, featuring
the catalogs of all three of the major global recording companies (Sony Music, Warner Music Group and Universal Music Group).

It quickly expanded the service to Australia, Mexico and Singapore, though without Universal’s catalog.

This past October, TikTok announced it would be holding its first-ever live music event, dubbed “TikTok In The Mix.”
</item 8>

<item 9>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
“This preference, combined with the high volume of music discovery and sharing activity on TikTok, can be harnessed to grow a fanbase without borders and has
helped to launch global artists from genres such as Reggaeton, Afrobeats and K-Pop.”

The report added: “TikTok users‘ interest in international music is evident in their music streaming preferences. Compared to the average music listener across
each market, this audience is more likely to indicate that ‘access to music by global artists’ is an extremely important factor when selecting a music streaming
service.”

“TikTok is already famous for being the launchpad for creating viral hits and breaking new artists, but we wanted to understand how TikTok and its users impact
the music industry more broadly,” Ole Obermann, TikTok’s Global Head of Music, said in a statement.
</item 9>

<item 10>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
SBF allegedly tried to buy off Trump: The TC team’s been trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced entrepreneur
accused of orchestrating the collapse of cryptocurrency exchange FTX. But fascinating details about SBF’s political dealings are emerging from a book by Michael
Lewis, “Going Infinite,” that debuted on the first day of the trial, like SBF’s attempt to buy off Trump to get him to not run again for president.

Gmail fights back against spammers: Google this week announced a series of significant changes to how it handles email from bulk senders in an effort to cut
down on spam and other unwanted emails. The company says that, starting next year, bulk senders will need to authenticate their emails, offer an easy way to
unsubscribe and stay under a reported spam threshold.

TikTok tests an ad-free tier: TikTok is testing an ad-free subscription tier for some users.
</item 10>

<item 11>
Twitch’s money guy talks about the revenue split controversy and its monetization long game (TechCrunch, 2023-10-26T22:28:10+00:00)
I have to wonder if those decisions to change these policies were in response to competition from other platforms.

I would say it this way: I think they are an acknowledgement, and maybe a closer connectivity to the community via Dan specifically. I credit Dan in his short
time as CEO to really make some big, hard decisions. Both the launch of Partner Plus, as well as simulcasting, the embed stuff we talked about in the keynote —
these are three things that I think are very much an acknowledgement of the feedback from the community. We’ve made that change, so I wouldn’t call it out of
competitive pressures. I would call it out of focus on the community.
</item 11>

<item 12>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 12>

<item 13>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
The attorney general’s press secretary blamed the site’s downfall on activists “hacking the system.”

The presidential election is more than a year away, but it may be too late for Ramaswamy to make any headway on TikTok, where the platform’s young users still
don’t take him seriously. He recently posted a video about getting ready for the next Republican debate, and was hit with yet another wall of edging comments.

Even if candidates like Ramaswamy did everything right — like having a platform that didn’t alienate young voters and working with creators with more favorable
reputations — they’d still represent a party that many Gen Z and millennial voters aren’t aligned with.

“It’s really hard to be effective with a generation when a lot of your policy is attacking them,” Henry said. “If all of your policy is highly unfavorable for
that generation, that’s a hard sell in itself, even if you’re an effective communicator with all the strategy in the world, to sell someone something they don’t
want.”
</item 13>

<item 14>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
The same month, it struck a deal with Distrokid to have indie artists upload their music to TikTok Music and to Capcut, TikTok’s video editing software, as well
as to TikTok’s Commercial Music Library, which offers a catalog of music for brands to use in their TikTok ads.

Meanwhile, TikTok parent ByteDance has launched an AI music-making app, and most recently, TikTok unveiled the “Add to Music App” feature, which enables TikTok
users to save music they’ve discovered directly to their Apple Music, Amazon Music (Prime or Unlimited) or Spotify accounts.

These new partnerships indicate that TikTok is moving away from what at times appeared to be a trying relationship with the music industry. In one notable
incident, TikTok disabled the use of copyrighted music for users in Australia, in what the platform described as an experiment, and what many observers saw as a
hardball tactic in its licensing negotiations with music rights holders.
</item 14>

<item 15>
VTubers have reached their inevitable conclusion (Polygon, 2023-10-19T16:01:50+00:00)
Brands, too: Remember the Tony the Tiger VTuber in August of 2022?

Many creators that jumped on the trend have since stopped using their avatars, but at the start of the COVID-19 pandemic, there was a large increase in VTubers
on Twitch and other streaming platforms, like YouTube, as models and equipment became more accessible to the general public. Twitch has since gone on to
highlight a wide variety of VTubers through its VTuber Takeover promotion, which most recently ran in August. Twitch also reported that VTubing-related content
on the platform shot up by some 350% between January and August 2022. Despite this, some have argued that the VTubing bubble has all but burst.

Polygon is celebrating TwitchCon 2023 with a package of stories covering streaming trends and news from the show. Check it out!

“I do feel that VTubing has become less novel as the market has become more saturated,” says a streamer with agency Nijisanji who is not approved to speak
publicly.
</item 15>

<item 16>
VTubers have reached their inevitable conclusion (Polygon, 2023-10-19T16:01:50+00:00)
“I think VTubing content has become kind of stale,” she says. She goes on to mention that a lot of VTubers tend to be boxed into specific kinds of content —
playing Minecraft or popular FPS games like Valorant. “In order for VTubing to continue to grow, VTubers need to branch out and create more content in line with
other influencers,” she says.

“Agencies should allow for talents to experiment, so long as it isn’t hurting anyone”

Mugi says that if VTubers took cues from other content creators and explored options like lore analysis videos or short-form reviews, the scene wouldn’t be as
stagnant. And like other VTubers, Mugi has dreams of going corporate. Already relatively successful, having created her own storefront that hosts collaborations
and merchandise featuring other VTubers and artists, she has auditioned for several major agencies. She says that one agency asked if she would close down her
business if she passed the interview phase and became one of its talents.
</item 16>

<item 17>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
A social social network network: Lauren reports that a former TikTok employee is building a social app for content creators to network and “spill the tea,” so
creators can help each other out making better, more engaging content.

X may go ad-free?: It appears that X (formerly Twitter) is testing three tiers of its Premium service, its CEO says. Under the hood, code shows one tier may be
ad-free. If it’s also troll free, please take my money right now.

Top reads on TechCrunch this week

A lot of amazing news on the site this week, but if we go by the raw numbers, here are the most popular stories — the ones that I didn’t already cover above,
that is.

Passwords? We don’t need no steenkin’ passwords: Passkeys are a phishing-resistant alternative to passwords that allow users to sign in to accounts using the
same biometrics or PINs they use to unlock their devices or with a physical security key. Google is now making it the default sign-in method for all users.
</item 17>

<item 18>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
One remaining – and major – potential pitfall for TikTok has to do with the rise in tensions between China and Western countries.

Amid concerns that parent company ByteDance may be sharing TikTok user data with the Chinese government, numerous governments – from the US to the European
Union to Australia – have banned the app from government-issued devices.

A small number of jurisdictions have banned or are attempting to ban TikTok altogether, including Montana and, most recently, Nepal.

Cinq Music Group’s repertoire has won Grammy awards, dozens of Gold and Platinum RIAA certifications, and numerous No.1 chart positions on a variety of
Billboard charts. Its repertoire includes heavyweights such as Bad Bunny, Janet Jackson, Daddy Yankee, T.I., Sean Kingston, Anuel, and hundreds more.Music
Business Worldwide
</item 18>

<item 19>
The new, ‘efficient’ Spotify has a very different approach to podcasting (The Verge, 2023-10-25T16:10:29+00:00)
CEO Daniel Ek echoed that sentiment. “We’re constantly finding new ways to bring more efficiencies out of the business… We’ve seen some improvements, but you
should expect us to continue to look for more improvements going forward because that’s just our modus operandi.”

While Ek and Vogel really hammered that “efficiency” theme (according to my transcript, they said the word more than 30 times over the course of the 50-minute
call), what was left unsaid about podcasting spoke louder. In the past, they would excitedly tout how many millions of podcasts were on the platform, the latest
new celebrity show, and the area’s explosive growth. But as investors have run out of patience for the time and money such ambitions required, the company’s
leadership changed its tune.

Rather than emphasizing, say, its new revenue sharing deal with Trevor Noah or the reliability of The Ringer, Ek pointed to Spotify’s new AI-driven translation
product.
</item 19>

<item 20>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
In the years since, Democratic campaigns have increasingly prioritized social media strategy, establishing politician-influencers who wield content for votes.

A 2022 midterm report by the Alliance for Securing Democracy found that in the Senate races, 47% of Democratic candidates had TikTok accounts, compared to 12%
of Republican candidates. Of the major party House candidates, 30% of Democrats and 10% of Republicans were on TikTok, which as a platform was most prevalent in
gubernatorial races.

Though opposition to TikTok has been bipartisan, the crusade to ban the platform over national security concerns is primarily led by Republicans at the state
level — a position that likely factors into the party’s social media struggles with young voters.

Mehmet Oz, one of the few Republican candidates on TikTok, had the advantage of already having a large social media following when he announced his candidacy
thanks to his popularity as Dr. Oz.
</item 20>

<item 21>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
That in and of itself is probably still largely unexplored territory that will see a lot of change and maturing in the next few years. One key aspect there
might be cost monitoring as well since companies like OpenAI charge largely “by the token,” which is a very different metric than most cloud computing.

What can be done to ensure industry-specific startups that tune generative AI models for a specific niche will prove defensible?
</item 21>

<item 22>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
For $4.99, subscribers get an ad-free experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that
it’s piloting the plan in a single, English-speaking market outside the U.S. for now.

LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it throughout; an AI-powered LinkedIn Learning coach; and a
new AI-powered tool for marketing campaigns.

Musk comes clean about X’s metrics — maybe: In September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million
posts every day, excluding retweets. But speaking at an event this week, X CEO Linda Yaccarino offered a contradictory figure.
</item 22>

<item 23>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
“Luminate’s findings prove what we had long known: that TikTok is the driving force behind music discovery in the industry, and that TikTok users are active,
engaged and highly valuable drivers of music industry revenues.”

The new study is part of an ongoing effort by TikTok to highlight its growing role in the development of global music culture. Last year, TikTok released a
report asserting that 13 out of the 14 songs to reach number on the US Billboard Hot 100 in 2022 were driven by viral trends on TikTok.

The data clearly shows why TikTok has focused so heavily on developing music-related services, to the point that one could be forgiven for thinking the company
plans to reinvent itself as a part of the music industry.

Last year, TikTok launched SoundOn, its own music distribution service, which has partnered with Believe-owned distro company TuneCore and Downtown
Holdings-owned B2B tech and services company FUGA.
</item 23>

<item 24>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
As the 2024 election approaches, Republican candidates may again fall behind in mobilizing voters on social media platforms.

Ramaswamy is also the only Republican presidential candidate to engage with TikTok, and is one of the few in his party to even have an account. His Silicon
Valley roots as a biotech entrepreneur and millennial upbringing set him apart from the other candidates, but he faces an uphill battle in convincing Gen Z to
like him, much less show up at the polls. Though his videos receive hundreds of thousands of views, his comments are also overrun with sex jokes.

Annie Wu Henry, the digital strategist behind Fetterman’s TikTok stardom, noted that younger voters who are very online tend to value raw, unfiltered
authenticity. They don’t see that in many older politicians, and especially not in the Republican party.

“Gen Z does not put up with bullshit,” Henry said.
</item 24>

<item 25>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.
</item 25>

<item 26>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
“In the global and US markets, TikTok user engagement metrics are strongly associated with streaming volumes. In other words, higher TikTok engagement — whether
that’s likes, views or shares — corresponds with elevated streaming volumes,” the Luminate report stated.

However, TikTok’s impact on music isn’t limited to DSPs. The analysis found that TikTok users are also larger spenders on music off-platform as well. By wide
margins, they are more likely to buy merch and attend live shows than the average consumer.

In the US, 45% of TikTok users bought music-related merch over a 12-month period, compared to 35% of overall music listeners. Meanwhile, 38% attended a live
music event during the year, compared to 33% of overall music listeners.

Overall, US TikTok users spent 22% more on music-related purchases, compared to overall music listeners. A similar pattern was found in the four other countries
they study looked at.
</item 26>

<item 27>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
A similar pattern was found in the four other countries they study looked at.

The study also found that the TikTok audience can be an effective marketing tool for artists, as TikTok users, in particular, are considerably more likely to
discover and share music than overall social media and short video platform (SFV) users.

In the UK, TikTok users were 108% more likely to discover and share music than other social media users, while in Germany, they were 100% more likely. US TikTok
users were 71% more likely to discover and share music.

“In doing so, TikTok users help both emerging and established artists to reach new audiences,” the Luminate report stated.

The study found that TikTok’s platform appears to be a major driver of the glocalization of music – the trend of local artists, performing in local languages,
being elevated to a regional or global platform.

“TikTok users have a stronger preference for international music than the average music listener,” the report said.
</item 27>

<item 28>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Do you expect startups to build striated solutions for individual layers of the LLM stack, or pursue a more vertical approach?

While there are startups specializing in parts of the stack (like Pinecone), Valor’s focus is on applied AI, which we define as AI that is solving a customer
problem. Saile.ai is a good example — it uses AI to generate closeable leads for the Fortune 500. Or Funding U using its own trained dataset to create a more
useful credit risk score. Or Allelica, using AI on treatment solutions applied to individual DNA to find the best medical treatment for you personally in a
given situation.

Companies like Datadog are building products to support the expanding AI market, including releasing an LLM observability tool. Will efforts like what Datadog
has built (and similar output from large/incumbent tech powers) curtail the market area where startups can build and compete?

Tools like Datadog can only help the acceptance of AI tools if they succeed in monitoring AI performance bottlenecks.
</item 28>

<item 29>
Gag City is a viral win for Nicki Minaj (TechCrunch, 2023-12-09T02:34:35+00:00)
Brand Twitter tends to turn fun trends into advertising opportunities, taking organic community interactions and spitting out contrived versions clearly made to
go viral. It may be grating, but in this case, it’s working in Minaj’s favor. This week, everyone wants to go to Gag City.
</item 29>

<item 30>
Gag City is a viral win for Nicki Minaj (TechCrunch, 2023-12-09T02:34:35+00:00)
Nicki Minaj is an artist who’s been embroiled in controversy throughout her career, from posting bad takes about COVID vaccines to defending her husband Kenneth
Petty, a convicted sex offender. She may be a brilliant artist, but her problematic history makes her far from the family-friendly public figure that brands are
more likely to endorse.

But with Gag City, Minaj has brands doing all of her marketing for her. “Pink Friday 2” is an artistic marvel in itself (though I am probably biased), but the
free promotion that it’s been getting as a viral meme is particularly astounding. Artists have spent the last few years trying to drum up engagement for their
work by making their songs trend on TikTok, which audiences have started to resist. Gag City doesn’t bank on being the viral song of the summer to drive
streaming numbers — the bit is removed enough for non-stans to enjoy it, while still revolving around the album it’s promoting.
</item 30>

<item 31>
Dropout’s dropping CollegeHumor on its 5th anniversary — here’s why and what the future holds (Polygon, 2023-09-26T17:35:00+00:00)
But we’ve given them names and voices. And now we’ll see! Now we’ll see how glib they are!

Reich: David Attenborough couldn’t do this!

Sam, that reminds me of something you said to NPR earlier this year, about how you found success with Dropout in, instead of going broad for the viral video,
going for “something that feels special to a small group of people.” Has five years of Dropout inspired any advice for creators, or producers, or fans —
basically anyone who’s looking to get away from algorithmically powered viral demand, to other platform structures?

Reich: My advice is find a better business model. I think we live in an era of online content right now that is being defined and shaped by algorithms. And
that’s true no matter the platform. I think different platforms are shaping content in different ways, and that’s kind of exciting as a creator. Like, Which of
these sports am I most excited to play?
</item 31>

<item 32>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
On Equity this week, the crew talked about the SBF trial; deals from VR firms Rainforest, At One Ventures, Section 32 and Greylock, where venture funding has
declined; and how Fearless Fund, a firm founded to invest in women of color, is being barred from awarding grants to Black women founders.

Meanwhile, Found featured Esther Rodriguez-Villegas from Acurable, a medical device company that makes patient-friendly wearable devices to diagnose and manage
respiratory conditions at home. As a career-long academic, Rodriguez-Villegas talks about how she never intended to be a founder until she learned about how the
currently available medical devices make it extremely difficult to detect and treat diseases like sleep apnea and epilepsy.

And over on Chain Reaction, Jacquelyn did a crossover episode with Alex about the SBF trial.
</item 32>

<item 33>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
He has also proposed barring anyone under 25 from voting, unless they serve in the military or pass a civics exam.

His first videos portray him as the one candidate who cares about America’s youth. While his Republican colleagues have largely shunned TikTok, Ramaswamy is
presenting himself as one of the Cool Politicians who will actually use it.

“We have a generation of politicians that is badly out of touch,” Ramaswamy said in his inaugural video, which did not describe his policy stances.

@jakepaul Getting Vivek on Tik Tok because i believe our politicians of the future should connect with gen z and milennials on social where we all live and
breathe. Its bizarre that in this day and age our presidents have no connection with us via social. Only the occasional tweets.
</item 33>

<item 34>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
To put into context the number of music subscribers at individual streaming services in the market, we reported in July that as of February, in the United
States, Spotify counted 44.4 million paying subscribers, while Spotify rival Apple Music had 32.6 million subscribers, and Amazon Music had 29.3 million
subscribers.

The new TikTok study looked at consumption data from Luminate Connect, audience insight from the Luminate Music 360 Study, and TikTok engagement volumes
provided by the platform in five markets: the US, UK, Brazil, Indonesia, and Germany, as well as globally (see below).

Including both paid and ad-supported streaming services, roughly nine out of 10 TikTok users in each market studied used at least one music streaming service.

In the US, 89% of TikTok users subscribed to at least one music streaming service, compared to 74% of overall consumers, while in the UK, 85% of TikTok users
had a streaming service account, compared to 62% of consumers overall.
</item 34>

<item 35>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
“Gen Z does not put up with bullshit,” Henry said. “That’s so much of what platforms like TikTok and Twitch cater to, and why they thrive on those platforms,
because you can just pull out your phone and be talking while you’re getting ready with the worst angle possible. Republicans in general, it goes against so
much of how they act otherwise and young people know that.”

Ramaswamy’s TikTok strategy is puzzling; he appears to be trying to replicate the success of previous candidates who became social media stars over the course
of their campaigns, but his TikTok presence conflicts with his own stances on social media and young voters. His communications director, Tricia McLaughlin, did
not immediately respond to TechCrunch’s request for comment.

Like many of his Republican colleagues, Ramaswamy has accused TikTok of being a threat to national security, and during a town hall days before he launched his
account, described the platform as “digital fentanyl” from China.
</item 35>

<item 36>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Only the occasional tweets. Meet @Vivek Ramaswamy ♬ original sound – GenosPicks

Establishing himself as the millennial politician who’s cool enough to use TikTok but anti-woke enough to play in the divisive culture war isn’t working in
Ramaswamy’s favor, though.

His account, which has amassed over 50,000 followers in the weeks since he joined, has been barraged with comments either criticizing his positions or trolling
him. He’s been the butt of Gen Z’s relentless comments about getting off to his content (the top comments on his videos are consistently jokes about edging). He
also incited the wrath of witchtok creators, who filmed themselves casting hexes upon him and other conservatives.

Ramaswamy is raking in TikTok engagement — even if it’s not how he intended — but Democrats are still dominating the Republican party in digital strategy.
Ocasio-Cortez’s Instagram Stories garnered her nationwide popularity when she campaigned in 2018, reaching further than her local constituency.
</item 36>

<item 37>
With Amo, the founder of Zenly wants to make social apps social again (TechCrunch, 2023-11-19T12:07:20+00:00)
With Amo, the founder of Zenly wants to make social apps social again The startup raised $20 million and is launching ID today

In 2018, I wrote a TechCrunch article stating that 2018 was “the year social networks were no longer social.” Reflecting on that article, I’m not sure that 2018
was the turning point. But the premise of the article still holds up well.

At some point, social networks were no longer about connecting with your closest friends, keeping up with long-distance family members and feeling a special
connection with people you love.

TikTok, Instagram, Snapchat, Facebook, YouTube and X (formerly Twitter) are all slowly evolving to become the same infinite scrollable feed of
algorithm-optimized short videos from top performers.

And it turns out that I’m not the only one who has noticed that social networks have been slowly drifting away from their original purpose.
</item 37>

<item 38>
Gag City is a viral win for Nicki Minaj (TechCrunch, 2023-12-09T02:34:35+00:00)
Welcome to Gag City, the pink metropolis inhabited by stans and brands alike.

In the days leading up to the release of “Pink Friday 2,” Nicki Minaj’s fifth studio album and sequel to her debut record “Pink Friday” that dropped on Friday,
Twitter was flooded with AI-generated images of pink-toned cityscapes. Gag City, the dreamy false utopia ruled by Minaj and her Barbz, broke through stan
Twitter and became a viral meme that brand accounts immediately used for their own marketing — promoting Minaj’s album for free.

Is it an authentic stan-led campaign to build hype for Minaj? Is it a plant to game engagement for both the album and brands? What’s clear is that the viral
moment is a win for Minaj, manufactured or not.

It started in September, when Minaj teased the album’s cover art online. The image features Minaj on a pink subway car, drifting through pink clouds with a
futuristic (and obviously, pink) city skyline in the background.
</item 38>

<item 39>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Does Reddit co-founder Alexis Ohanian have regrets over his creation? The entrepreneur and founder of venture fund Seven Seven Six, speaking at an event at the
University of Virginia, espoused a fairly negative view of the current social media landscape and the creator economy — an environment where he says, everyone
has “been so trained and conditioned now to just get the likes and retweets, get the upvotes…I wish I could be more optimistic about it, but we’re seeing it
play out before our eyes.” Still, he ultimately described himself as a “techno-optimist” — an apparent nod to a16z General Partner Marc Andreessen’s recent
diatribe where he claimed technology could solve all our problems.

“Guilty as charged,” Ohanian said, during his interview with Kara Swisher. “I do think that pendulum will swing back. I think even swifter than we realize.”
</item 39>

<item 40>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
Spotify erased thousands of AI-crafted songs from its platform but also recently globally launched an AI-powered DJ that curates music for listeners while
talking to them in a synthetic voice.

“The training of generative AI using our artists’ music . . . begs the question as to which side of history all stakeholders in the music ecosystem want to be
on: the side of artists, fans and human creative expression, or on the side of deep fakes, fraud and denying artists their due compensation,” Universal Music
Group said after a song using AI to imitate Drake and The Weeknd, two of its artists, went viral.

These same conversations and contradictions are manifesting across creative industries, but artists themselves don’t always have a seat at the table.
Independent artists in particular are learning that their voices resonate louder when coming together across disciplines to push back against what Holland
describes as an “extraordinary spectrum of exploitation” that leverages their work.
</item 40>

<item 41>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Republicans still don’t know how to talk to young voters online Vivek Ramaswamy struggles to gain an edge in the face of TikTok trolls

In an appeal to younger voters, Republican presidential candidate Vivek Ramaswamy — who proposed raising the voting age to 25 — launched his TikTok presence
with an endorsement from Jake Paul, the YouTuber turned boxer who built a content empire by marketing to children.

Ramaswamy is one of the only Republican politicians making an effort to connect with Gen Z and young millennials, a demographic that overwhelmingly supported
Democrats in the midterm elections. Despite the popularity and growing influence of far-right creators online, Republican candidates have historically failed to
engage young voters on social media, if they try at all.

While Democratic politicians like Rep. Alexandria Ocasio-Cortez and Sen. John Fetterman’s social media campaign strategies set the blueprint for politicians
seeking internet fame, Republicans have neglected to adopt the digital fluency required to secure the youth vote.
</item 41>

<item 42>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Third-party Reddit app Narwhal says it hopes to survive Reddit’s app purge with a subscription plan.

Apropos “getting together,” Amanda reports that VidCon is still kickin’. For the first time, the conference hosted an industry leadership summit, where creator
economy experts and creators could hash out their grievances with the state of the business and share ideas to make the job of a creator more sustainable. That
makes sense, unlike creators raising venture capital: It’s so eye-wateringly hard to make money as a creator, I’d love to see the pitch that convinces a VC to
cut a check to a creator, with a realistic expectation of a venture-scale return.

Maybe they were hiding behind the sofa?: Sarah reports that Mastodon actually has 407K+ more monthly users than it thought — and it seems like Twitter has a lot
more traffic than Musk said. Still, the peak now is about the same as it was a decade ago, and it’s unclear what the social platform can do to encourage more
growth.
</item 42>

<item 43>
Gag City is a viral win for Nicki Minaj (TechCrunch, 2023-12-09T02:34:35+00:00)
), the Empire State Building and countless others posted their versions of Gag City.

On one hand, memes tend to die the minute brand accounts start co-opting them — nothing is more tiresome than seeing a fun joke turn into a corporate-friendly
marketing ploy. AI-generated images are already ethically fraught, and critics have raised concerns over AI generators trained on artwork without the consent of
the work’s artists. Artists have also criticized brands for using AI-generated art instead of commissioning work from a real, human artist. Though using
AI-generated art for commercial use is legal, as copyright laws pertaining to AI are virtually non-existent, it’s generally seen as a shitty move by many in the
art world.

On the other, it’s free promotion for Minaj, and as a lifelong Barb who spent her adolescence running a stan account for “Pink Friday,” I consider it a win.
</item 43>

<item 44>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hiya, folks, and welcome to Week in Review (WiR), TechCrunch’s digest of the past week in tech news. It’s TC’s column that highlights the major stories over the
past few days, and — we humbly submit — it’s a darn useful resource for folks on the go.

This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing
FTX fallout. Also on the agenda: Gmail’s harsher rules to prevent spam, TikTok testing an ad-free subscription plan, and LinkedIn going big on AI tools. And
that’s not all.

If you haven’t, sign up here to get WiR in your inbox every Saturday. And if you have, our thanks. Now, let’s get on with the news.
</item 44>

<item 45>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
“The amount of media and content that we consume, political and cultural, where people get their information … It’s getting more and more intertwined,” Henry
said. “To be effective ideologically, when it comes to campaigns and these wins that we want, we have to understand how these ecosystems work, and be
strategically using them or else we risk being left behind.”

Engaging with voters online requires more than just working with popular creators. In Ramaswamy’s case, an endorsement from Jake Paul only drove animosity from
TikTok viewers. While his older brother, Logan Paul, has made the occasional social commentary on his podcast, the younger Paul has rarely engaged with politics
or social issues. Jake Paul is generally disliked online for the running list of allegations against him, including sexual assault accusations.

The Paul brothers have a reputation of promoting anyone and anything for a paycheck, and when Jake Paul posted a political endorsement out of the blue, viewers
immediately questioned how much Ramaswamy paid him.
</item 45>

<item 46>
Twitch’s money guy talks about the revenue split controversy and its monetization long game (TechCrunch, 2023-10-26T22:28:10+00:00)
There are a couple of things that I’ll point out. One is acknowledging what Dan has done as the CEO to get out and engage with streamers, hear their concerns,
and really influence our priorities as a company, is really a consequence of him both being an active streamer as well as being extremely engaged within the
community.

And then, more broadly for both Jeremy [Jeremy Forrester is VP of Community Product] and myself as the product leaders, building the product is continuing to
innovate and ensure that we are absolutely a unique service. Whether that’s features, like we’re working on Stories, whether that’s monetization — it’s all in
service of ensuring that livestreaming creators can be more successful on Twitch.

For a long time, there was a sort of exclusivity with Twitch, where there were exclusive deals, and simultaneous streaming on other platforms was forbidden. I
have to wonder if those decisions to change these policies were in response to competition from other platforms.
</item 46>

<item 47>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Before Rogier launched MasterClass, he worked as a VC, and — through his connections — he received a $500,000 seed round before he even had an idea for a
company.

And on Chain Reaction, Jacquelyn interviewed David Pakman, managing partner and head of venture investments at CoinFund. Before CoinFund, David spent 14 years
at the venture capital firm Venrock. He also led the Series A and B rounds at Dollar Shave Club, which was acquired by Unilever for $1 billion. And, in 1991,
David co-created Apple Music when he was part of Apple’s system software product marketing group.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
</item 47>

<item 48>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
White House deputy chief of staff Jen O’Malley Dillon told Axios that the administration is trying to not only reach young people, but “people whose main way of
getting information is digital.”

“If you’re not going to be directly on the platform, having surrogates — whether those be influencers, celebrities, normal people — do the messaging, that’s
likely going to be reaching people at a more personal level anyway,” Henry said.

“Everyone has a vested interest, for the most part, in what the president of the United States has to say, but if it’s your friend, if it’s this person you’ve
followed for five years, you have a vested interest that’s a little bit more personal. Usually that’s more effective.”

The “influencer army” strategy could be legally and ethically murky when influencers are paid to spread political messaging, potentially skirting both federal
campaign ad laws and platform rules. TikTok bans political ads, and in recent elections, cracked down on posting sponsored political content.
</item 48>

<item 49>
62% of TikTok users in the United States pay for a music streaming service (report) (Music Business Worldwide, 2023-11-21T22:03:00+00:00)
MBW’s Stat Of The Week is a series in which we highlight a data point that deserves the attention of the global music industry. Stat Of the Week is supported by
Cinq Music Group , a technology-driven record label, distribution, and rights management company.

TikTok is in the process of reinventing itself as both a partner to music companies, and a competitor to streaming and distribution services.

Now, a new market analysis sheds some light on why the ByteDance-owned short video platform has focused so intensely on expanding its services in the music
world, with the firm reporting to be a major driver of music consumption and spending.

While that may not come as a surprise to those who’ve noted the overlap between TikTok and music culture, the numbers in the analysis shine a light on the scale
of TikTok’s influence.

According to the analysis, commissioned by TikTok and carried out by US market monitor Luminate, 62% of US TikTok users pay for a music streaming service,
compared to 43% of all consumers.
</item 49>

<item 50>
Twitch’s money guy talks about the revenue split controversy and its monetization long game (TechCrunch, 2023-10-26T22:28:10+00:00)
It’s a very reliable source of income for many streamers.

So I think the combination of making it easier to run, increasing the rev share and then recently in the keynote, we talked about two new features to allow
moderators to know when an ad break is coming, and adjust the ad break around the content because the problem in livestreaming, as you’re aware, is that people
don’t want to miss out. So we’re very focused on improving the ad experience by allowing the ad breaks to be synchronized with the content and enrolling
moderations and eventually the community in adjusting those ad breaks.

Like you said, livestreaming is very different from static content, and tactics for monetization that work for Instagram or TikTok just don’t really work for
Twitch or any livestreaming platform. In those cases, what has Twitch done differently from those more standard platforms in terms of making monetization more
seamless?
</item 50>
=== END ITEM id=7135fb5a974aa21c ===

=== ITEM id=cf4cad947255822a ===
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
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.
</item 1>

<item 2>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
It was something that I’ll never forget.” Loading Australia, who lost last year’s final to Canada, has enjoyed a soft run on paper to this year’s decider,
eliminating the Czech Republic and Finland – with neither country boasting a top-30 singles player – but will have earned their 29th Davis Cup title if they
account for Italy. Canada defeated Italy in the Davis Cup round-robin matches in September, but bowed out to Finland in the knockout stage without Felix
Auger-Aliassime and Denis Shapovalov, who have both battled injury and form. The final is due to start at 2am AEDT on Monday and will be broadcast on 9GEM.

Sinner ends Djoker’s Davis Cup dream Novak Djokovic’s dream of capping his great season by blasting Serbia past Australia in the Davis Cup final was blown up by
Jannik Sinner’s extraordinary double triumph over the world No.1.
</item 2>

<item 3>
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
</item 3>

<item 4>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Following their 2-0 victory over Finland in Friday’s semi-final, executed superbly without a dropped set in singles wins for Alexei Popyrin and Alex de Minaur,
team captain Hewitt said he couldn’t be prouder of their achievement in making back-to-back finals.

And even though they’ll be facing either a Serbian side led by world No.1 Novak Djokovic or an Italian team inspired by young superstar Jannik Sinner in
Sunday’s (Monday AEDT) final, the inspirational de Minaur was adamant: “No one really wants to play us.” He delivered once more for Australia, taking an 11th
victory in his last 13 cup singles matches, sealing Friday’s (Saturday AEDT) win by defeating Emil Ruusuvuori 6-4, 6-3. Alex de Minaur stretches to return the
ball against Finland’s Emil Ruusuvuori.
</item 4>

<item 5>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Instead, in the Davis Cup final in Malaga, Australia will face an Italy team running hot on the heroics of their golden boy Sinner, who beat Djokovic in both
the singles and the doubles in the space of five exhilarating hours in their semi-final on Saturday (Sunday AEDT). Jannik Sinner saved three match points before
going on to beat Novak Djokovic. Credit: Getty Images In one of the great Davis Cup rubbers, Djokovic had held three match points against Sinner and was on the
verge of a completing a relatively workaday 2-0 victory for the Serbs.

But for only the fourth time in his unparalleled career, Djokovic was beaten from match-point up as Sinner roared back to complete his second victory over the
master in just 12 days, winning 6-2, 2-6, 7-5.
</item 5>

<item 6>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Yet the 22-year-old proved absolutely nerveless, firing down an ace to save the third of the three match points, and then immediately turning the tables by
breaking Djokovic in the very next game amid wild Italian celebrations in the stands at the Palacio de deportes Martin Carpena.

Sinner then served out for a famous win, ending Djokovic’s cherished unbeaten 21-singles winning streak in the Davis Cup and becoming the first man since 2019
to beat the Serb in singles twice in the same year. The drama between the two rivals wasn’t over, as the two teams inevitably both drafted in their top singles
players into the doubles decider. Sinner was again magnificent and still looking ominously fresh after more than four hours on court, with Djokovic’s weak
backhand into the net off the Italian’s second serve in the doubles on match point seeming to sum up his disappointment. The beneficiaries of all this?
</item 6>

<item 7>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
Three looming ODIs against India in the next week and three ensuing T20s will be the first showing of that, with the hosts also a rising force in the white-ball
game.

Advertisement

“Within our group we’re really comfortable with where we’re at, and the progression we’re on,” Healy said.

“Ultimately there’s a big T20 World Cup in Bangladesh towards the end of next year, that is a real carrot for us.

“These are the sort of big trophies that we’re wanting to win.

“And yes, every series in between now and then is important, but at the same time we’re building towards that World Cup.

“Maybe you can ask me that (supremacy) question at the end of the six white-ball games to see how we’re going and we can answer that.

“These next six games will probably give us a better indication about how we’re tracking but also what we might need to improve on.”
</item 7>

<item 8>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The beneficiaries of all this? Lleyton Hewitt’s Australian side, who were happily enjoying a day’s rest at their team hotel further down the Costa del Sol,
watching the drama unfold on TV.

While the Italy tie didn’t finish until 8.15pm after a noon start, Alex de Minaur and co should be fresh for combat in Sunday afternoon’s (Monday AEDT) final as
they seek to lift the Davis Cup for the first time in 20 years. AAP De Minaur, Popyrin power Australia into Davis Cup final Lleyton Hewitt has declared his
close-knit tennis band of brothers will give it “a hell of a shake” to bring back the Davis Cup to Australia after they roared into another final in Malaga.
</item 8>

<item 9>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Loading Hewitt animatedly embraced Popyrin after his straight-sets victory over 171st-ranked Finn Otto Virtanen, but Australia’s Davis Cup captain would be even
more thrilled if he could repeat the dose against Italy. Australia could also turn to an alternative No.2 singles option, with Jordan Thompson and Purcell
waiting in the wings. “It’s everybody’s dream to play for their country,” Popyrin said. “I looked up to Lleyton, looked up to all the older Aussie legends. To
be able to put on the green and gold, to be able to go out there and play for your country, no matter the format, if it’s this Davis Cup or the old Davis Cup,
you give your all.

“It’s nerves, it’s pressure – you’re playing for everybody back home. It’s something that nothing will ever compare to ... it really was the biggest moment of
my career to get the first live rubber win, to get one foot into the final.
</item 9>

<item 10>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
The current tour of India is not a multi-format series, but the defeat in the one-off Test marks Australia’s first loss in a four-day game since January 2014.

But quizzed over whether the defeat suggested Australia’s unprecedented period of dominance was ending, Healy was defiant.

“It’s probably not even a blip,” Healy said.

Advertisement

“It (Test cricket) is a format where we’re not overly familiar with, in conditions that we’re not familiar with.

“I see more positives out of this and than any sort of negativity.

“There is always a lot of expectation and external noise about our team and how we’re performing.

“They are so used to seeing us winning, that I think, at times, they forget that we’ve actually seen quite a lot of change over the last 12 to 18 months.

“That kind of goes unnoticed external to our group.”

Australia have spoken about shifting their mindset in white-ball cricket after the losses in England, pursuing a more attacking approach.
</item 10>

<item 11>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
De Minaur, cheered on by his excited mum, then didn’t need similar heroics to Wednesday’s as he overcame the loss of the first two games.
</item 11>

<item 12>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
“Hopefully we can go one better this year,” he added, recalling the 2-0 defeat by Canada in 2022. De Minaur had been the hero on Wednesday, pulling out an
extraordinary comeback win over Jiri Lehecka with Australia on the verge of defeat in their quarter-final with the Czech Republic.

But he was happy to pass the laurels to Popyrin, a late squad replacement who’d been asked to take on the troublesome “second singles” role by Hewitt. Loading
“Alexei hasn’t played for Australia in a very long time [two years], had a massive match today against a very high-quality opponent and dug super deep,” said de
Minaur. Popyrin managed to deliver a rock solid game, making only 11 unforced errors to his erratic opponent’s 25, to prevail in just under an hour-and-a-half.
“It was probably the biggest win of my career,” he admitted.
</item 12>

<item 13>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Fans of Australian rules football in the Harbour City have a reputation for being fickle. After a winless campaign in 2022, the Swans – led by ex-Collingwood
player Chloe Molloy – have been competitive this season. The team won their first AFLW match in round one and are within a game of the top eight with two rounds
to play.

But general manager of women’s football at the AFL, Nicole Livingstone, said the Swans’ off-field offering had tapped into something “uniquely Sydney” and
helped make matches more attractive.

“To go to North Sydney Oval and to have quince paste and hand-picked grapes, cheddar cheese and biscuits in a little box that’s got this beautiful wrapping on
it – they’ve really thought about what they’re doing and made it an experience,” she said.
</item 13>

<item 14>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The 22-year-old Sinner, by now an irresistible force, then re-emerged to team up with Lorenzo Sonego and spray massive forehand winners around liberally as they
outplayed a by now weary-looking Djokovic and Miomir Kecmanovic 6-3, 6-4. Djokovic had three match points in the singles rubber. Credit: Getty Images It was a
crushing end to Djokovic’s hopes of ending his triumphant season, which has featured three grand slam victories and an ATP finals showdown victory over Sinner
just six days ago, with a second Davis Cup title. After telling fans to “shut up” earlier in the week, he again got irritated with the crowd in the doubles,
pretending to be the conductor to the Italian fans’ deafening chorus, but though he may be the greatest player who ever picked up a racquet, doubles looks an
alien game to him.
</item 14>

<item 15>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But crowds are short of targets that could accelerate the extension of the regular season, which grows from 10 games this year to 12 games by 2025.

5,474 fans were at North Sydney Oval to watch the Swans’ first AFLW victory earlier in the season. Photograph: Jack Snape/The Guardian

The league enjoyed strong attendances in round one, which were partly attributed to the so-called Matildas effect, but some clubs have struggled to attract fans
in large numbers this season. Crowds of less than 1,500 have been recorded in both traditional markets such as Melbourne and Perth, as well as areas targeted
for growth like Blacktown and the Gold Coast.

The A-League Women launched this month with a first round highlighted by a crowd of more than 11,000 in Sydney.
</item 15>

<item 16>
Tennis News: Kyrgios escalates beef with Becker, change ‘warrior’ Novak had to make, WTA criticised for charisma fail (The Roar | Sports Writers Blog,
2023-12-19T23:54:26+00:00)
I hope that one day we will be closer, to be able to sit and talk and reflect. It would be truly incredible.”

Hewitt at the helm

Australia’s David Cup captain Lleyton Hewitt will return to captain the Aussie team at the United Cup.

Eighteen nations will contest the tournament in Perth and Sydney starting December 29 and ending on January 7.

Advertisement

The tournament began last year with Hewitt and Sam Stosur at the helm.

David Witt, who helped lead the United States to the 2023 trophy, also returns in 2024 for the Americans’ title defence.

Can Djokovic go the distance?

Former Spanish star Alex Corretja has questioned Novak Djokovic durability ahead of a Golden Slam attempt in 2024.

The Paris Olympics adds an extra high profile tournament to the four majors next year.
</item 16>

<item 17>
‘So used to seeing us winning’: Healy denies Aussie era ending after India loss, calls for more Tests (The Roar | Sports Writers Blog,
2023-12-24T22:28:25+00:00)
Alyssa Healy has rejected any suggestion Australia’s stranglehold on international women’s cricket is slipping, saying their Test loss to India is barely a blip
on the team’s record.

Australia stumbled to their first Test defeat to India by eight wickets in Mumbai on Sunday, after being outplayed by the hosts in foreign spinning conditions.

The loss comes after one of the more difficult periods for the Australian side in the past six years, with the retirements of Meg Lanning and Rachael Haynes.

Facebook Twitter Whatsapp Reddit Email Share

After winning their fourth straight global ICC event earlier this year in the 50-over World Cup, Australia retained the Ashes via a drawn multi-format series in
England in July.

That tour included losing four of six white-ball matches, with Australia’s win in the Test enough to have them level on points with England and hold onto the
trophy.
</item 17>

<item 18>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Credit: AP It had followed a superb win from teammate Popyrin, who defied jangling nerves to earn a 7-6 (7-5), 6-2 victory over Otto Virtanen. It meant the
Australians’ crack doubles team of Max Purcell and Matt Ebden weren’t even needed.

It will be Australia’s 49th appearance in the Davis Cup final, as they seek the trophy for the 29th t28-time champions Australia, in a 49th final, seek a first
win in the World Cup of men’s tennis in 20 years. “We’re a very solid squad, a lot of depth in every single position, and if we can get ourselves to a deciding
doubles, we’ve got an unbelievable pairing as well,” de Minaur said. “So we singles guys have just got do our best to try and get a win and let the doubles guys
do the job.
</item 18>

<item 19>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The second of Sinner’s Djokovic victories came in Italy’s impressive Davis Cup semi-final triumph over Serbia, after saving three consecutive match points on
serve, before besting the world No.1 again in the doubles. World No.4 Sinner, coached by Australia’s Darren Cahill, is the first player to topple Djokovic twice
in the same season since Russia’s Daniil Medvedev in 2019. Italian teammates Lorenzo Sonego and Jannik Sinner will face Australia in the Davis Cup final.
Credit: Getty However, if Popyrin can get the job done in the No.2 singles then Australia will, at minimum, reach the doubles with a live chance, and a good
one, with 2022 Wimbledon champions Max Purcell and Matt Ebden to be unleashed.
</item 19>

<item 20>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Quince paste and hand-picked grapes have played a part in helping the Sydney Swans launch themselves to the top of the AFLW attendance charts, less than two
years after their first match in the women’s league.

The Swans play their final home match of the regular season this weekend with an average attendance of 4,365 from four games so far – 124 more than the next
best club Adelaide – and a marquee pride match against Collingwood at Henson Park to come on Sunday where a crowd in excess of 4,000 is expected.

The club’s focus has been to target local communities and make match days compelling for fans on and off the field, according to executive GM of marketing at
the Swans, Loretta Johns.

“Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting
our games,” she said.

Fans of Australian rules football in the Harbour City have a reputation for being fickle.
</item 20>
=== END ITEM id=cf4cad947255822a ===

=== ITEM id=5399938e2e1b7b86 ===
Gold sentences and facts:
Sentence 0: The present Board of Trustees of the Fund which consists of the Secretary of Education as Chairman with the Director-General of the National Economic and Development Authority, the three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of the Philippines (CEAP) and the Philippine Association of Colleges and Universities (PACU) shall also constitute the proposed Board of Trustees of the PhilEd.
  Fact 0.0: The present Board of Trustees of the Fund consists of the Secretary of Education as Chairman.
  Fact 0.1: The present Board of Trustees of the Fund includes the Director-General of the National Economic and Development Authority.
  Fact 0.2: The present Board of Trustees of the Fund includes three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of the Philippines (CEAP), and the Philippine Association of Colleges and Universities (PACU).
  Fact 0.3: The present Board of Trustees of the Fund shall also constitute the proposed Board of Trustees of the PhilEd.
Sentence 1: In addition, Andrew Wilkinson, 72, has informed the HSF board of his intention to retire as a board director, which will reduce the company’s board to five directors.
  Fact 1.0: Andrew Wilkinson is 72 years old.
  Fact 1.1: Andrew Wilkinson has informed the HSF board of his intention to retire as a board director.
  Fact 1.2: Andrew Wilkinson's retirement will reduce the company's board to five directors.

Retrieved items:
<item 1>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
Likewise the role of PhilEd would be to fund the education dreams of all Filipinos especially the neediest and most deserving.

Filipinos have three dreams in education. The first is to enroll in a chosen course in the school of their choice. This dream will be funded by the School
Vouchers Program which will now be transferred from DepEd to PhilEd.

The second dream, especially for Filipino parents, is to save enough money to fund the college education of their children, hence the prevalence of pre-need
Educational Plans. Unfortunately, most of the pre-need companies went into bankruptcies causing financial ruin and emotional distress to their plan holders.
There has been a massive market failure and so government must intervene. But intervention should not involve the government offering educational plans but
rather in insuring these plans. Just as the Philippine Deposit Insurance Corp. (PDIC) insures the bank deposits of Filipinos, so will the PhilEd insure their
educational plans.
</item 1>

<item 2>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
The reason for the delay? According to UK-listed HSF, it’s all to do with a recently-updated company valuation from HSF’s official independent valuer,
US-headquartered Citrin Cooperman (CC).

The HSF board says this valuation from CC was “materially higher than the valuation implied by proposed and recent transactions in the sector”.

HSF’s board particularly points to two transactions/would-be transactions as evidence:

The proposed sale earlier this year of assets to Blackstone-backed Hipgnosis Songs Capital for USD $440 million (pre-costs) – or $417.5 million (post-costs) – a
proposal that was ultimately rejected by HSF shareholders; The sale of 20,000 “non-core” songs from HSF’s portfolio to an unnamed buyer (possibly Kobalt Music
Group) for $23.1 million, as announced on December 11.
</item 2>

<item 3>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
HSF also confirmed a number of other new measures in a ‘circular’ to its shareholders today.

These measures included the fact that, if Hipgnosis Songs Fund shareholders vote to remain a publicly-traded fund (via that 2023 ‘continuation vote’) at the
firm’s next AGM, then another ‘continuation vote’ will be put in front of HSF shareholders at an Extraordinary General Meeting in January 2026.

(The ‘continuation vote’ that will take place in Q4 2023 has been scheduled ever since HSF went public in 2018 – i.e. five years after it floated. The 2026
‘Continuation Vote’, as newly-agreed, means another one is now being scheduled for around two years’ time.)

In addition, HSF today confirmed some new amendments to its ‘Investment Advisory Agreement’ (IAA) with HSM.
</item 3>

<item 4>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Brown noted that a 0.5% rise in Citrin Cooperman’s discount rate (i.e. raising it from 8.5% to 9.0%) would “reduce the fair value of the [HSF] portfolio by
7.9%, which on a leveraged basis is [approximately a] 10% reduction in NAV [Net Asset Value]”.

Should HSF’s board go down this route, he noted, he expected “the [HSF] shares to weaken… given that this represents another ‘unforced error’”.

“This is an early blow to the credibility of the new [Hipgnosis Songs Fund] Board, and casts further doubt over the credibility of the independent valuer,
Citrin Cooperman.”
</item 4>

<item 5>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
If a ‘superior proposal’ (versus the $440 million offer from HSC) is received during the ‘go-shop’ period, said HSF today – whether or not HSC uses its
‘matching right’ to beat it – it may be necessary to adjourn the HSF AGM (and therefore the 2023 ‘continuation vote’) into November.

In its note to shareholders today, the HSF board reiterated its belief – and Hipgnosis Song Management’s belief – that HSF owns “a unique portfolio of iconic,
culturally significant songs that will deliver strong long-term value as they benefit from the structural tailwinds in the music industry”.

The note added: “Furthermore, the [HSF] Board believes that the Investment Adviser’s approach to Song Management should enable the Company to outperform the
wider music market.
</item 5>

<item 6>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Not to mention this line in various HSF company reports to date: “The [HSF] Board is ultimately and solely responsible for overseeing the valuation of the
Company’s investments in music catalogues and has appointed [Citrin Cooperman] to perform this specialist work.”

WHAT’S really going on?

It’s a fair bet that Hipgnosis Songs Fund’s board – which last week appointed activist investor Christopher Mills as a non-exec Director – is making something
of a ‘hidden’ statement to shareholders with today’s announcement.

It may be a coded attempt by HSF’s board to gnash teeth towards HSM/Mercuriadis’ ‘call option’ – which, as MBW has pointed out before, leaves the exec with the
exclusive power to sweep in and acquire HSF’s portfolio under several different circumstances.
</item 6>

<item 7>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
Just like PDIC which is funded from the premiums paid by the banks, PhilEd will fund this program from the premiums by the educational plan companies. PhilEd
will be ideally suited to do this as it could swap the failed educational plans with school vouchers from its repository of school vouchers.

The third dream is based on the realization of Filipino school graduates that their learning must continue even as they leave their alma maters. For in this
modern age, the fount of personal knowledge must continuously be replenished. In healthcare, we have the Health Maintenance Organizations (HMOs) which
continuously monitor and minister to our health. We need Lifelong Learning Organizations (LLOs) to continuously monitor and minister to our learning. HMOs exist
in the Philippines. LLOs do not. In partnership with the international development agencies and the private sector, PhilEd will create this industry. The
international development agencies will provide the technical assistance grants that will create the developmental and regulatory framework for the industry
while the private sector will organize the LLOs.
</item 7>

<item 8>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
its share price).

For example: On September 14, HSF’s market cap stood at approximately GBP £1.125 billion, which represented a 40% discount on the firm’s operative NAV of USD
$2.316 billion as of the end of March this year.

(Said ‘operative NAV’ was calculated by an independent valuer – Citrin Cooperman – using a discount rate of 8.5%.)

This is where the importance of that potential $440 million transaction comes in.

As MBW reported earlier this month, Hipgnosis Songs Fund’s shareholders are currently considering a proposal to sell 29 of the company’s catalogs to another
Hipgnosis company – the Blackstone-backed Hipgnosis Songs Capital (HSC) – for USD $440 million.
</item 8>

<item 9>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
Even so: one wonders if some new changes at HSF, announced today, are related to this ‘continuation vote’ – and whether or not they amount to concessions to
help keep HSF investors happy and confident in the firm’s long-term prospects… and voting ‘yes, let’s keep going’.

The shareholder frustration: NAV vs. public market

First, a bit of important context.

The fundamental revenue performance of HSF doesn’t seem to be under scrutiny from shareholders: the company posted its best-ever revenues in the 12 months to
end of March, while streaming ‘pro-forma’ revenue was up 14.8% YoY in calendar 2022.

Where HSF shareholders – as well as its management – are finding frustration is in the current gulf between the firm’s ‘operative Net Asset Value’ (‘operative
NAV’) and its value on the public market (i.e. its share price).
</item 9>

<item 10>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
That price, according to HSF’s board, represented a 14.2% discount on Citrin Cooperman’s valuation of this portfolio as of September 30 (something we’ll come
back to).

As part of its announcement this morning, the HSF board took something of a swipe at its investment adviser, Merck Mercuriadis‘ Hipgnosis Song Management.

After seeing Citrin Cooperman’s latest valuation and believing it to be too high, said HSF, it then “sought advice from Hipgnosis Song Management… which is
majority owned by funds managed and/or advised by Blackstone, on their opinion on the independent valuer’s valuation.”

HSF’s board added: “Hipgnosis Song Management… eventually provided an opinion, which was heavily caveated, such that the Board has concerns as to the valuation
of the Company’s assets in its interim results.”
</item 10>

<item 11>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
There are a few clues in HSF’s latest public statement to this end, perhaps including its rare – not to mention conspicuous – mention of HSM being
“majority-owned by funds managed and/or advised by Blackstone“.

In addition, today’s statement is also a clear signal from the HSF board that it has understandable concerns over the now-infamous ‘discount rate’ used by
Citrin Cooperman to value HSF’s portfolio (as MBW covered in depth through here).

Citrin Cooperman continues to stick to an 8.5% discount rate in its valuations, despite interest rate rises over the past 12-18 months – and has long refused to
move this 8.5% figure upwards to, say, 9% or 9.5%.

If Citrin Cooperman ever did raise this discount rate, the independent valuation of Hipgnosis Songs Fund’s portfolio would, in turn, instantly fall.
</item 11>

<item 12>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
“[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try and pin the
blame on [Hipgnosis Songs Management].” Christopher Brown, JP Morgan

Concluded Brown: “[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try
and pin the blame on [Hipgnosis Songs Management].

“Our view is that a breakdown in the relationship between Board and manager would not be optimal for maximising shareholder value.”

Brown further noted that JP Morgan would “not be surprised to see Citrin Cooperman resign, having been very publicly undermined by the [HSF] Board today”.

JP Morgan retained its ‘overweight’ rating against HSF’s share price “in spite of yet another [HSF] own goal”.
</item 12>

<item 13>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
A Final thought…

As mentioned in the opening sentence of this article, there is plenty of drama surrounding HSF right now – from the ‘discontinuation vote’ we saw at the company
in October, through to a recent lawsuit launched against Hipgnosis Songs Fund by a former business partner of Merck Mercuriadis.

Yet, amongst all of the noise, what will likely most interest HSF’s investors (including the likes of Investec Wealth & Investment, Aviva Investors, and
BlackRock) is the ongoing underlying commercial performance of the firm’s copyrights.

How are Hipgnosis Songs Fund’s assets performing vs. the market? How is the company’s ‘premium-only’ acquisition strategy holding up against its promise of
growth “uncorrelated” to other industries? And what does all of this suggest about the ongoing growth potential of the HSF portfolio vs. the multiples that
Merck Mercuriadis paid to acquire it?
</item 13>

<item 14>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
That potential transaction is being managed for both HSF and HSC by two separate teams within their mutual investment advisor, Hipgnosis Song Management (HSM),
which is run and part-owned by Merck Mercuriadis.

When it announced its intention to explore the $440 million catalog sale to HSC two weeks ago, Hipgnosis Songs Fund confirmed that one of its motivating factors
was that the transaction could potentially act “as a catalyst for a re-rating of [HSF’s] share price”.

Aka: Increasing that share price to bring it up closer to the current operative NAV of the company.

HSC’s $440 million offer to acquire the 29 catalogs, HSF confirmed earlier this month, equates to a 26% increase on the amount HSF paid for those catalogs to
acquire them in the past.
</item 14>

<item 15>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
One factor potentially making the future termination of HSM as HSF’s investment advisor less likely? MBW understands that, in such a scenario, HSM would have a
first-refusal option to acquire HSF’s copyright portfolio – and could potentially turn to Blackstone (i.e. the backers of Hipgnosis Songs Capital) to find the
money to pull off such an acquisition.

Why the $440 million offer is connected to the date of HSF’s ‘continuation vote’

While HSF shareholders are considering the $440 million offer from HSC to buy the 29 catalogs, other non-Hipgnosis companies are also being invited to make bids
for the same portfolio.

Those bids are enabled by a ‘go-shop’ clause in the HSF/HSC proposal that allows the HSF board to consider these non-Hipgnosis approaches for a limited period
(until 11.59pm on October 23, to be precise).
</item 15>

<item 16>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
These amendments would give HSM a 12-month notice period as HSF’s investment advisor should the IAA be terminated at any point following the upcoming 2023
‘continuation vote’.

Another clause in the new IAA: If HSF’s share price stands at an average discount to ‘operative NAV’ of 10% or more across the month of January 2025 (with that
NAV determined at the time of publication of HSF’s interim report for the period to September 30, 2024), the HSF board says it intends to serve notice to
terminate the IAA with HSM.

However, HSF’s board has also given itself the freedom to un-terminate the IAA during the entire notice period that would follow (e.g. if HSF’s board serves
12-month notice on HSM at the end of January 2025, it would have until the end of January 2026 to withdraw its termination decision).
</item 16>

<item 17>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
In response, Hipgnosis Song Management issued its own statement today: “Hipgnosis Song Management has fulfilled its duties to [HSF] with respect to both the
independent valuation and preparation of the interim results in a timely and efficient manner.

“Notwithstanding the Board’s decision to delay publication of the interim financial statements, [we] will continue to work in a constructive manner to support
the interests of the Company and its shareholders.”

The oddest thing about all of this? Citrin Cooperman’s valuation of Hipgnosis Songs Fund is, according to HSF’s own company documentation, appointed and
overseen… by the HSF board.

See this line, in every annual and interim report from HSF so far (bolding MBW’s own): “Portfolio Independent Valuer… Citrin Cooperman Advisors LLC, formerly
Massarsky Consulting, Inc., [is] appointed by the Board to independently value the Company’s Catalogues within the Portfolio.”
</item 17>

<item 18>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
In a rather damning note issued today covering the delay in the publication of HSF’s results, Brown wrote: “This is an early blow to the credibility of the new
[Hipgnosis Songs Fund] Board, and casts further doubt over the credibility of the independent valuer, Citrin Cooperman.”

Brown added that he was “surprised” to see that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results,
especially as Kroll was appointed to consider the “reasonableness” of Citrin Cooperman’s assumptions for HSF’s previous set of results (to end of March 2023).

Added Brown: “We think the easiest solution now for the [HSF] Board is to apply an additional discount rate to the [Citrin Cooperman] valuation to reflect
current market uncertainty, on the basis that it is probably too late to employ another valuer.”
</item 18>

<item 19>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
the multiples that Merck Mercuriadis paid to acquire it?

Such questions will only be answered when Hipgnosis Songs Fund’s board elects to release the company’s six-monthly financial results to end of September 30.

The most important KPI within those results, in MBW’s view: the company’s Pro-Forma Revenue, and how it compares, like-for-like, vs. the previous year’s
equivalent period.

In the meantime, back in the ‘real world’ of the music biz, HSF is actually enjoying quite a fanciable Q4.

The company is benefitting from its cuts on hit albums including Taylor Swift’s 1989 (Taylor’s Version), via its acquired Jack Antonoff catalog, as well as
Swift’s Lover (via its acquired Joel Little catalog).
</item 19>

<item 20>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
The $440 million price also reflects a +51% premium on the value of the 29 catalogs implied by Hipgnosis Songs Fund’s public market cap for the 30 days ending
September 13.

What happened today?

Amongst the news confirmed by HSF today (September 28) is that the firm’s Chair of its board since 2018, Andrew Sutch, 73, is stepping down, both as HSF board
Chair and as a director of the company.

HSF says it’s begun a process to recruit a new Chair; Sutch (pictured inset) will leave his post as and when this person is hired, or in any case before HSF’s
AGM next year (in Q4 2024).

In addition, Andrew Wilkinson, 72, has informed the HSF board of his intention to retire as a board director, which will reduce the company’s board to five
directors.
</item 20>

<item 21>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Worth remembering: The HSF board has known about Citrin Cooperman’s latest valuation (as of September 30) for at least the past eight days.

We know this because, as mentioned, on December 11, HSF announced that the $23.1 million price of an agreed HSF ‘non-core’ asset sale represented a 14.2%
discount vs. Citrin Cooperman’s latest valuation of the portfolio (i.e. the HSF board had seen this latest valuation).

Yet it took until today (December 19) for HSF to alert its investors to its dissatisfaction with this valuation.

WHAT does JP Morgan make of it all?

One of the closest watchers of Hipgnosis Songs Fund in recent years has been JP Morgan and its analyst Christopher Brown.
</item 21>

<item 22>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
Today (September 28), HSF informed its investors that “credible third parties are already engaged in this go-shop process” following its announcement on
September 14.

As we’ve previously reported, there are potential obstacles in the way of these third parties’ bids. These include the fact that HSM (on behalf of HSC) has a
‘matching right’, meaning that if HSC decides to match the higher bid of a rival, HSF has to sell the catalogs to it – rather than the rival.

In addition, any ‘superior’ offer would have to exceed the aggregate cash net proceeds that would ultimately reach HSF’s coffers should HSC’s $440 million offer
be executed. (Read more about this condition through here.)

Still, it’s a theoretical possibility that a non-Hipgnosis company swoops for the 29 catalogs from HSF, and gazumps HSC in the bidding process.
</item 22>

<item 23>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
The present Board of Trustees of the Fund which consists of the Secretary of Education as Chairman with the Director-General of the National Economic and
Development Authority, the three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of
the Philippines (CEAP) and the Philippine Association of Colleges and Universities (PACU) shall also constitute the proposed Board of Trustees of the PhilEd.

The management team of PEAC will be the incoming management team of PhilEd.

Interestingly, PhilHealth, in addition to premiums collected from members as well as subsidies from local governments who have enrolled indigents in the
program, receives a share of the “sin taxes” collected from Tobacco and Sugar Sweetened Beverage. The justification for this is that tobacco smokers and sugar
sweetened beverage drinkers impose an additional burden on the healthcare system of the Filipinos and so must contribute to their health services. In Fiscal
Year 2022, this amounted to P74 billion.
</item 23>

<item 24>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
PhilEd will be the “bastonero” or project manager, managing the project as well as mobilizing government support and assistance.

FAPE or the Fund for Assistance to Private Education was a part of the Special Fund for Education set aside by the American government in 1963 from the surplus
funds authorized by the War Damage Act of 1962 and organized under Executive Order 156 issued by President Ferdinand Marcos on Nov. 5, 1968.

From the initial seed money of $6,154,000 or P215 million (based on the 1968 peso to dollar exchange rate), the FAPE fund in 55 years had barely grown to P277
million. In contrast the trustees managing the fund have expanded their services, basically as service provider to the DepEd. As a consequence, it has been spun
off as a separate organization, the Private Education Assistance Committee (PEAC).
</item 24>

<item 25>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
In Fiscal Year 2022, this amounted to P74 billion.

Moreover, under the Universal Health Care Law, PhilHealth receives 50% of the government share from the income of the Philippine Amusement Gaming Corp. (PAGCOR)
as well as 40% of the Charity Fund of the Philippine Charity Sweepstakes Office (PCSO). In 2022, this amounted to P188 billion.

In like manner, in addition to the school vouchers funded under the Expanded Government Assistance to Students and Teachers in Private Education Act, we suggest
that PhilEd be allocated a share of the taxes imposed on our extractive industries. This is based on the argument that the depletion of our natural resources
can only be justified if the taxes collected will enhance our human resources.

Like PhilHealth, PhilEd could also be mandated to receive a percentage from the collection of documentary stamps which have recently been raised.

Finally, PhilEd can be given access to the Special Education Funds of the local school boards of the Local Government.
</item 25>

<item 26>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
Hipgnosis Songs Fund (HSF), the UK-listed music rights fund, looks likely to hold its all-important 2023 ‘continuation vote’ on Thursday, October 26, according
to an announcement from the firm today.

Why’s that ‘likely’ rather than ‘definitely’? Because there’s still a possibility it might get bumped into November – the reasons for which we’ll get into later
in this article.

The ‘continuation vote’ will take place at HSF’s Annual General Meeting (AGM), and will see the firm’s shareholders vote on whether to keep the entity going as
a closed-ended investment entity i.e. a publicly-traded bundle of music rights.

How will that vote go? Obviously we don’t know yet – and there’s the small matter of a $440 million transaction looming that could potentially influence things.
</item 26>

<item 27>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
WHAT’S HAPPENED?

MBW Explains is a series of analytical features in which we explore the context behind major music industry talking points – and suggest what might happen next.
MBW Explains is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

The drama at Hipgnosis Songs Fund isn’t letting up for the Holidays, it seems.

This morning (December 19), the company’s board – now led by Chairman, Rob Naylor – issued a statement announcing that HSF would no longer, as previously
expected, be issuing its latest financial results today.

Instead, the results (covering the six months to end of September 2023) are now expected to be published sometime before the close of December 31.

The reason for the delay?
</item 27>

<item 28>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
This has been evidenced by the 44% total return, including right to income, transaction fees and expected taxes, realised since acquisition on the [$440
million] Transaction.

“Furthermore, the Board and the Investment Adviser are committed to ensuring that this value is achieved for shareholders by a re-rating of the share price and
are determined to deliver on the ongoing opportunity of the Company.”Music Business Worldwide
</item 28>

<item 29>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Christopher Brown, JP Morgan, on the HSF board statement today (December 19)

In addition, Brown expressed his surprise that HSF asked HSM to “opine on the valuation given [HSM’s] conflict of interest”.

His point: If HSM turned around and said, “yep, HSF’s valuation is 25% too high, better bring it down”, then it would produce a new valuation for HSF that may
allow Blackstone (or the Blackstone-controlled Hipgnosis Song Capital) to launch an acquisition bid for the HSF portfolio. (Or, at least, a part of that
portfolio.)

The likely price for this hypothetical Blackstone bid would be smaller than the failed $440 million ($418.5 million post-costs) bid that HSC (via HSM) made for
a chunk of the HSF portfolio earlier this year.
</item 29>

<item 30>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
We recently presented a paper at the 5th Paderanga-Varela Memorial Lecture entitled “Alternate Service Delivery Systems: Health and Education.” In that lecture,
we looked at the Department of Health (DoH) and the Department of Education (DepEd) as service delivery systems, delivering healthcare and education
respectively.

We concluded that the service delivery system of the DoH is far superior to the DepEd for two main reasons. The first is that the DoH has devolved its hospitals
to the local government units while the DepEd has not. Secondly, the DoH has spun off its health financing into the Philippine Health Insurance Corp.
(PhilHealth) while the DepEd has not. In this article we argue for the creation of the Philippine Education Development Fund (PhilEd).

When Isidro Consunji, President of DMCI Holdings, Inc. was asked by his officers what the role of his Chief Financial Officer (CFO) would be, he gave a
three-word answer, “Fund my dreams.”
</item 30>
=== END ITEM id=5399938e2e1b7b86 ===
