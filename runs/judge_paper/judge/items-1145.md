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

This file has 1 items.

=== ITEM id=8b76f6051dc8ce5d ===
Gold sentences and facts:
Sentence 0: "We all found ourselves at home looking at the people we worked with, the people we went out for drinks with and the people in our extended family on a computer screen," he told Fox News Digital.
  Fact 0.0: A speaker said that they all found themselves at home looking at the people they worked with, the people they went out for drinks with, and the people in their extended family on a computer screen.
  Fact 0.1: The speaker told Fox News Digital this.
Sentence 1: We used to play in person, but since the pandemic I run the game virtually online from my house.
  Fact 1.0: The speaker used to play the game in person.
  Fact 1.1: Since the pandemic, the speaker runs the game virtually online from their house.

Retrieved items:
<item 1>
CEO David Baszucki’s mission to make Roblox a billion-player platform (The Verge, 2023-10-12T14:01:00+00:00)
Jay Peters: Hi. Jay Peters with The Verge. You mentioned walled gardens in terms of PlayStation and Xbox, but I kind of feel like Roblox is its own walled
garden. We see lots of other shared virtual experience apps like Fortnite or Meta’s Horizon Worlds. Is there any chance or any thinking on some kind of
interoperability between all that stuff?

DB: I think there are two types of interoperability. One would just be on valuable items like a pair of shoes from Nike and whether the NFT supports the
interchange. That may not be a 3D graphical technical spec; that may be a little bit more of a cloud ownership spec. I think that this genre is evolving so
quickly to network hundreds of people to have realistic 3D. Are my shoes ultimately made of leather? Do they actually bend and flex? The technology is going to
go so quickly here that any 3D interchange is going to be almost like a five-year-old file format.
</item 1>

<item 2>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
The gaming industry, which likes to boast about how much money it makes in comparison to other forms of entertainment, is laying off workers nearly every week
at top studios. Meanwhile, the overall gaming market is shrinking. The cracks in the system that produces and promotes big-budget games are getting hard to
ignore.

“I wouldn’t trade it for the world,” Tenney said in his goodbye video, which is overflowing with footage of esports casters and viewers in awe of his Fortnite
plays.

“I’ve been doing this since I was a kid, man,” he continued, shaking his head. “I feel like my childhood and even some of my adulthood kind of flushed away. I
feel like this is the end of the journey, man.”
</item 2>

<item 3>
TwitchCon 2023 was better than last year, but everyone hated Las Vegas (TechCrunch, 2023-10-26T00:15:43+00:00)
To its credit, Twitch is receptive to community feedback, and has had to dramatically scale up its conventions as the platform’s usership skyrocketed since the
pandemic started in 2020. Attendees complained that last year’s event, which was the first convention that Twitch hosted in North America in nearly three years,
was overcrowded and poorly laid out. The San Diego Convention Center’s tight hallways couldn’t accommodate so many people, which was worsened by fans flocking
around popular streamers and blocking the flow of traffic.

Attendees also complained that the venue was inaccessible for those using mobility aids, such as wheelchairs and walkers, because of the unmanaged crowds and
unprepared security. In an X thread about the event’s lack of accessibility, attendees said that when popular streamers walked by, swarming fans pushed around
those using mobility aids and nearly trampled them.

Twitch managed to mitigate crowding issues during this year’s convention, a success that Delphin credits to the company learning quickly.

“It was a huge demand to come back.
</item 3>

<item 4>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
But as for social media as a whole, of which Reddit would presumably play a part, Ohanian said it brings out the “worst parts of our nature in many ways.”
What’s more, he expressed worries about the desire for people to be first and win on whatever leaderboard a platform features, which then leads to a situation
where misinformation is able to spread.

Humans evolved trusting the faces & voices we knew around the campfire. The idea of believing something bc of experts/institutions you've never met is actually
new to our species. I visited @UVADemocracy + chatted with @karaswisher + @dkroy about what's next for social media. pic.twitter.com/PMBXi7T3vv — Alexis Ohanian
🇦🇲 (@alexisohanian) October 31, 2023

“Now, there’s so much pressure to be first, and UGC — user-generated content — will always win. It won’t always be right, though.
</item 4>

<item 5>
CEO David Baszucki’s mission to make Roblox a billion-player platform (The Verge, 2023-10-12T14:01:00+00:00)
I think you made a prediction at RDC a couple of weeks ago about dating. You think people are going to be dating in Roblox.

It was really fun when we made this prediction because we were very careful on that slide. And then you saw the 17-plus people ID validated in 17-plus
experiences.

So parents don’t have a heart attack.

Yeah, yeah, yeah, given our history of safety and civility and the focus on that. But ultimately, what’s really interesting about the dating market, there’s
probably a third of the population that won’t go on Bumble or Tinder or Hinge or whatever just because of awkwardness. In an immersive 3D avatar-type
communication where you can be Shrek and I can be whoever I want to be, there’s actually a certain way to... I’ll be Donald Duck.

Okay. [Laughs]

There’s actually a little bit of a breakdown of the friction, of the fear of a video call, so I actually do believe that’ll happen.
</item 5>

<item 6>
Fortnite is the latest game to entice players with a portal to the past (TechCrunch, 2023-10-30T19:18:39+00:00)
Competition is stiff though, more so now that Nintendo just casually dropped Super Mario Bros. Wonder, the first two-dimensional side-scrolling Mario game in
more than a decade — another portal to the past given its departure from a long string of recent 3D Super Mario titles.

Square Enix was so confident in its remake of the 1997 Japanese roleplaying mega-hit Final Fantasy VII that the reimagined version of the game comprises three
standalone full-length games, each spaced a few years apart.

World of Warcraft paved the way

Fortnite’s experiment to bring players back into the fold by relaunching an original version of the game will be interesting to watch, but it isn’t a first.
World of Warcraft, the fantasy roleplaying game once synonymous with online gaming, did something similar just a few years ago. Blizzard Entertainment, which
developed the massively multiplayer online game (MMO), launched World of Warcraft Classic back in 2019, even as a modern version of the game remained online and
playable.
</item 6>

<item 7>
CEO David Baszucki’s mission to make Roblox a billion-player platform (The Verge, 2023-10-12T14:01:00+00:00)
We’ve always used the term “human co-experience” or “bringing people together.” The metaverse context, as we know, was coined with the book Snow Crash a long
time ago. And it’s interesting. It’s gone and ebbed through various flows. But I guess we imagine this more as a communication and connection platform. We’ve
never really used the term metaverse a lot, and I think going forward, we’ll probably always think of ourselves as a communication and connection platform.

Alex Kruglov: Hi, my name is Alex Kruglov. I run a startup called pop.in, which is also in the social gaming space but for a much older user base. I’m curious:
when you think about 24 hours in a day and you talk about the future that you see with a billion daily actives and all kinds of time spent not just playing
games but anything from dating to shopping and so on, what are people not going to be doing and switching?
</item 7>

<item 8>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Thankfully, in the doldrums of 2020, someone I knew knew someone who knew someone (that’s literally what the connection was) who was interested in running a
game. That kicked off my D&D career, but was this year that everything finally clicked in my brain and it went from a hobby I enjoyed to one that I spend every
free minute of my spare time thinking about.

Part of it is finally having reliable groups. After some initial shuffling of members, in 2023 I’ve been blessed to have not one but two separate D&D groups,
both with enthusiastic members who are committed to at least trying to meet regularly. And yes, part of it is Baldur’s Gate 3, which helped me to understand
some of the more complicated D&D mechanics and just generally made me very excited about combat in a way I never was before.

I’ve still yet to actually have a game in person! My groups are spread out across North America, but I’m hoping to manifest an in-person session soon.
</item 8>

<item 9>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Triangle Agency

“I laughed so hard my face hurt.” —me, 10 minutes after my first session of Triangle Agency

I love a game with style and a clear, fun point of view, so I’ve been a fan of Triangle Agency since I picked up its player guide in beta form earlier this year
(to prep for writing a published adventure for the game). Written in the form of a cheerful, slightly threatening employee handbook, the Field Agent Manual
guides you through the creation of your supernaturally powered character, then gets you ready for your role finding and containing Anomalies that threaten the
nature of reality. After your team’s morning meeting, of course.

Beyond the style of the materials, though, playing the game was an absolute blast. The mix of mundane life, weird occurrences, and corporate BS created a
fast-paced game experience that was playful and easy to understand (just count those threes!
</item 9>

<item 10>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
For me, Project ECCO found that balance.

A story of time travel and cosmic horror, Project ECCO is easy to get into and relatively quick to play (three to 10 hours per game). Between the hundreds of
narrative prompts, unlocking various time travel “devices” (coins, dice, tarot cards, etc.), and the absolutely stunning layout, this is one of my favorite
TTRPGs, period.

Played in the pages of a yearly planner, Project ECCO hits every beat of a good time-travel story: time loops and timeline divergences, a shady and controlling
Time Travel Agency, and an unknowable cosmic Entity consuming spacetime. As you play, you (literally) burn through the days of your planner, creating an
artifact, marked and changed by the telling of your agent’s story.

For an example of gameplay and a masterfully produced actual play, listen to the Project ECCO miniseason of My First Dungeon with game designer Elliot Davis.
</item 10>

<item 11>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The whimsically illustrated rulebook lays out the tools by which you’ll lay bricks for your city, weaving together your ideas and your fellow players’ to
establish Landmarks, Residents, and Neighborhoods, all on a quest to find the city’s true names. It’s a beautifully simple game, guided by a Compass, which can
be anything — literally. It’s almost impossible for me to capture the magic of this game with words, because each session we played was so entirely unique, from
the verdant turtle-city in the sky we built out in stages to the haunted shell of a city we later used as our D&D setting. It’s a perfect game full stop, but
especially for those seeking to rekindle their creativity, or simply to sit down and create with friends.

Triangle Agency

“I laughed so hard my face hurt.”
</item 11>

<item 12>
TwitchCon 2023 was better than last year, but everyone hated Las Vegas (TechCrunch, 2023-10-26T00:15:43+00:00)
“I think most of us agree TwitchCon Vegas felt very positive in a way that hasn’t been felt in some time,” Lowco posted on X (formerly Twitter). “But also we
are positive we don’t want TwitchCon to be in Vegas again. Lol.”

It’s the first North American TwitchCon to be held outside of California since Twitch’s inaugural convention in 2015. Las Vegas may be able to physically
accommodate such a large event, but many TwitchCon attendees complain that it lacks the infrastructure to foster the community engagement that Twitch revolves
around. Daytime hangouts outside of the convention center were challenging; spaces that don’t revolve around drinking or gambling are scarce on the Strip, and
if they do exist, they’re too far from the convention center to quickly visit.

“The convention center in Vegas was solid but the city had way too much going on,” Lowco added. “We don’t need the distractions of Vegas, there’s enough for us
to do.
</item 12>

<item 13>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
Things haven’t changed much since 2019 at some major studios: Diablo 4’s developers say they also underwent crunch to ship the game. Naturally, the action
role-playing game had Twitch integration from the jump, and alongside a detailed content roadmap. There’s little incentive for any of this to change, even as
big-budget games buckle under their own weight and studios are closing left and right. Not when publishers know that the longer a player is hooked, the more
likely it is that they’ll spend money.

At this point, live-service games need content creators like Twitch streamers to stay solvent. According to the developer speaking to Polygon, streamers help
old games transform into new and visible moments. Developers need live-service games to stay in the conversation, because the base experience can worsen without
a persistent player base.

It’s “not even from a cost perspective,” the developer says, though obviously more players make profits more viable. “But [if] we don’t have 20,000 players,
then matchmaking times are terrible.
</item 13>

<item 14>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Critics on both the right and the left described the launch as a disaster.

Conservative organizations have proven especially out of their depth when facing the wrath of extremely online social media users, who mobilize trolling for
social justice. In 2020, TikTok users reserved hundreds of tickets for a Trump rally and never showed up, leaving the president to face swaths of empty seats.
When a Texas anti-abortion group opened an anonymous tip form to enforce the state’s ban on abortion after six weeks, Gen Z activists flooded the site with
Shrek porn. This year, TikTok and Twitter users shut down the Missouri attorney general’s tip form for reporting clinics that provide gender-affirming care.
Within days of launching, the site was spammed with fanfiction, satirical anecdotes about kids getting “transed” and fanfiction. The attorney general’s press
secretary blamed the site’s downfall on activists “hacking the system.”
</item 14>

<item 15>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
For the first year in recent memory, scale did not necessitate tradition, and scope did not preclude getting weird. In 2023, nothing was sacred in video games,
and so they felt more vibrant than ever.

Sure, some of the more “focused” games threw us for a pleasant loop: Dredge begins as a lonely fishing sim before transforming into something otherworldly, and
Humanity morphed from a pensive art project into an all-out war. Dave the Diver, similarly, is not so much about being a diver as it is about running a sushi
restaurant, or hunting for alien artifacts, or conversing with said aliens, or — you get the point. Whether you booted up your Steam Deck for a cross-country
flight or hid your Switch off screen during that boring Zoom meeting, the game you returned to was rarely the one you left behind.

This amorphousness (I’m begging our copy editor to let this one slide, because what other “word” could adequately summarize the video games of 2023?)
</item 15>

<item 16>
Dropout’s dropping CollegeHumor on its 5th anniversary — here’s why and what the future holds (Polygon, 2023-09-26T17:35:00+00:00)
[laughs] Adam Conover, who is one of my dear, dear friends and also remains pretty mixed about digital versus traditional entertainment, he’s not sure where he
wants to fall in terms of those those relative opportunities. He’s like, The thing about digital is you create for yourself a jail, and then you have to live in
that jail forever, because your audience wants that from you, again and again and again. Whereas in traditional media, you have a certain amount of seasons, and
maybe you get canceled, then you move on to something else.

I think Brennan and I both are in jail right now. The way in which we’re lucky is we both jailed ourselves inside anthology series.

Mulligan: Yeah.
</item 16>

<item 17>
Twitch is the best way to watch sports, when you can actually watch sports on Twitch (Polygon, 2023-10-20T17:30:00+00:00)
He adds that it’s been interesting seeing fans evolve from those that were fans of his gaming streams to ones who are sports fans, or discovering gaming fans
who are also sports fans.

“I have people that have been watching me for years playing World of Warcraft or doing something that isn’t football, but they’re actually huge football fans,”
Esfand says. “A big part of Twitch is that people like watching streamers they can relate to. Because what happens is, it’s not just about the streamer; it’s
also about the streamer’s chat. Now they are getting involved in the community, whether it’s just through chat itself or through Discord.”

Alex Casassovici, founder and CEO of Azarus, a company that works with online creators to enhance the streaming experience, says that livestreaming sports and
sports-related content on platforms like Twitch creates a more intimate experience for streamers and fans alike.
</item 17>

<item 18>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
After more than a decade of going on and on about the “board gaming renaissance,” it’s safe to say that tabletop role-play is finally having its moment in the
spotlight. But while Dungeons & Dragons’ OGL fiasco has clearly lit a fire under many players eager to make a change, the fact of the matter is that a sizeable
cohort of writers and designers have been toiling away at excellent games for years now. Their tireless efforts have yielded a bumper crop of excellent, some
might say genre-defining, TTRPGs. Just like board games, a few modern classics have also cropped up: Look no further than The Quiet Year and The One Ring, both
making their second appearance on this list.

Just as before, Polygon asked nearly two dozen writers, designers, presenters, actors, and personalities from around the world of tabletop gaming to share with
our readers the TTRPGs that made an impact on them and their players this year. Here’s what we found.
</item 18>

<item 19>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
Many would say the writing is on the wall for Twitch, after endless headlines about losing top talent, poor management, bad platform policies, and creator
burnout. But the idea that Twitch has lost its way after an ostensible golden era is an easy narrative that misses something much starker. If Twitch is in
trouble in 2023, it’s not because things have gotten worse. Everything you’re seeing is the natural progression of an unsustainable system. The prognosis has
always been terminal; we’ve just refused to see it because it’s been profitable.

It didn’t take long for publishers to see that Twitch could make or break a game. Around 2018, Twitch and its personalities turned Fortnite into a juggernaut so
powerful, much of the gaming industry started molding itself in its image. While an approachable aesthetic and surprising gameplay made for good viewing, what
kept Fortnite from being a mere fad was the model. Before, content expansions and adjustments could take months, if not years, to hit, and you might have to
shell out money to see any of it.
</item 19>

<item 20>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
Instead, she pivoted to being more of a lifestyle and beauty personality who uploads on short-form platforms like TikTok, if and when she feels like it.

The evolution grew from a similar motivation as Tenney’s. In the video where she explains it all, Anys said that during her time as a full-on Twitch streamer,
keeping up with the “rat race” often meant playing what was popular while being unable to do “basic human things,” like going grocery shopping. How could she,
when streaming all day was the norm and there was always competition around the corner that was younger and more willing to stay live for longer?

It was inevitable, then, for things like sleep streams, uncapped subathons, and randomly generated AI shows to materialize on Twitch; content creators feel like
they need to keep going, always. Barring that, live-service games are always at the ready. In her sort-of-goodbye video, Anys spoke frankly while quickly
scrolling through her entire game history on Twitch. It was all live-service games like League of Legends and Fortnite.
</item 20>

<item 21>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
As far as games go, there are probably very few out there that require such a strict fidelity to the road laid out before you, but City of Winter uses this
structure to facilitate the kind of deeply resonant character arcs and development that many role-players crave in their games. An astoundingly complex game,
bringing your best stories to the table will leave you with a sense of loss and belonging that seem to walk together, hand in hand.

Dead After Dinner

Jason Morningstar, owner, creative director, and lead designer at Bully Pulpit Games

Jenn Martin’s game Dead After Dinner combines the relentless procedural revelation of For the Queen with Knives Out-style murder mystery, and it is endlessly
hilarious. Everyone is part of a miserable family stuffed with resentment, somebody is getting killed, there will be uncomfortable questions, and a murderer
will definitely be revealed. Or will they? Dead After Dinner is a truly pick-up-and-play murder mystery that will delight you and your scenery-chewing friends.
</item 21>

<item 22>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
CAN YOU AVOID END-OF-THE-YEAR CAREER BURNOUT? JOB EXPERTS REVEAL SECRETS

Corcoran agreed, noting that many report feeling livelier, happier and more active during in-person meetings.

"Because video platforms artificially limit our perspective, we miss the cues we’re used to from live, in-person interactions," she told Fox News Digital.

"As social beings, we’re highly attuned to these cues, and it takes more work for us to look for and process them in the virtual environment, leading to the
fatigue we experience."

Even beyond the workplace, Elliott noted that people are using video in more personal contexts than ever before, such as sharing reels on social media and
calling loved ones on FaceTime.

"Because video platforms artificially limit our perspective, we miss the cues we’re used to from live, in-person interactions."

"This can further exacerbate the problem if you're already spending a significant portion of your day on Zoom," he said.
</item 22>

<item 23>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
What’s more, Larian had two years of player feedback with which to build its new masterpiece. That masterpiece launched in August, and it garnered the kind of
attention that catapults a game from niche interests to widespread acclaim.

Old CRPG fans finally came home, the Mass Effect generation has discovered a new kind of RPG to sink its teeth into, and D&D fans have found an exceptional
digital version of their beloved tabletop game to play alone or with their party in co-op.

Like a good DM, Baldur’s Gate 3 teases out story beats that feel personal to you, cleverly luring you into experiences that seem like they wouldn’t make sense
in anyone else’s campaign. Every act delivers quiet, memorable character moments and anxiety-inducing battles. All of these situations feel organic, allowing
Baldur’s Gate 3 to replicate, perhaps as closely as a video game can, Dungeons & Dragons’ best feature: the intoxicating sense that anything could happen at any
moment.
</item 23>

<item 24>
CEO David Baszucki’s mission to make Roblox a billion-player platform (The Verge, 2023-10-12T14:01:00+00:00)
Clay Christensen has this concept of jobs to be done. Who are they firing? What jobs are they firing in order to be on Roblox?

I asked this from the point of view of a parent of three kids, the middle of whom — for all intents and purposes, it’s crack when it comes to Roblox. It’s
highly, highly regulated by her parents.

AH: Yeah, how dystopian is this going to be?

DB: Yeah. We’re actually very optimistic. And I think we’re optimistic because, in the specter of social media, there’s a wide range of things that people do on
social media. Some consume a lot of short video content and trigger dopamine. Some are “compare my life with your life and accelerate FOMO.” Some are hopefully
similar to the feeling of, “Well, my kid’s on the phone with their friends. They’re connecting, inventing, hanging out.”
</item 24>

<item 25>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
I played a one-shot with some friends I was reconnecting with and we mostly group game mastered. We played a gang of messy screw-ups who didn’t hugely get
along, my character had traded her old gender to the fae over the summer, and we tried to investigate a potential monster hunter, a NPC most of our characters
found too cool to talk to. There was a gag about someone doing a club remix of “Danny Boy” at a memorial service and the group chat with those friends is still
named “Danny Boyz (The Pipes, The Pipes).”

Mörk Borg

Jasmine Bhullar, writer, content creator, executive producer of DesiQuest

Of the games I delved into this year, although several afforded hours of entertainment, I felt myself repeatedly returning to one in particular. Although Mörk
Borg was released in 2020, it was only in 2023 that I happily stumbled upon it and was able to somehow get a table together to run a game.
</item 25>

<item 26>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
I’ve rolled for every character, grinded out all of said characters, 100%-ed the exploration, completed every quest — did all the stuff that would burn anyone
out. Honkai: Star Rail is Hoyoverse’s answer to the Genshin Impact burnout.

The slower-paced turn-based combat, the ability to auto-battle (if your teams are strong enough), and the smaller maps make it the perfect daily game for me to
funnel my interest into without feeling exhausted. The characters are fun, flashy, and easy to latch onto. Should I main the sneaky but elegant Kafka, who uses
damage-over-time skills to whittle her opponents away? Or should I use the brooding Blade, who unleashes huge attacks at the cost of his HP? Ah, I guess I’ll
let my wallet decide — it is a gacha game, after all. —JL

20.
</item 26>

<item 27>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Here’s to 2024 being an even better D&D year for me.

Eating Oranges in the Shower

Sam Dunnewold, Dice Exploder podcast

The Golden Cobra challenge is always a deep well of exciting (and free!) new larps from the bleeding edge of the hobby. This year, like a song that’s stuck in
my head, I haven’t been able to stop thinking about Eating Oranges in the Shower by Hazel Anneke Dixon, a game about exactly what it sounds like. You play as
members of a group chat who discover the /r/ShowerOrange subreddit, a real community dedicated to the “liberating” act of eating an orange while taking a
shower. You each decide to try this out for yourself before returning to compare notes, and because it’s a LARP, you actually go do the thing.

This game is so strange and delightful, not to mention delicious. The principal act is compelling on its own. It feels forbidden somehow, even as it’s obviously
harmless.
</item 27>

<item 28>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—Pete Volk

Related Arcane is officially League of Legends canon now

41. Goodbye Volcano High

Developer: KO_OP

Where to play: PlayStation 4, PlayStation 5, and Windows PC

Goodbye Volcano High is a visual novel with rhythm game elements, and it takes place at the precipice of the end of the world. It’s centered on a group of
teenage dinosaurs entering their senior year of high school, a perfect balance between high school dramatics and the grim future of Earth, as a meteor rockets
toward the planet. Though the rhythm game elements can feel a bit finicky — and don’t seem to matter much, in terms of progressing the game — Goodbye Volcano
High’s music only adds to the dimensional, raw experience created by worker-owned studio KO_OP.

It’s rare to find a game that takes the teenage experience seriously, but Goodbye Volcano High does just that. It’s a time in your life where you feel so, so
much.
</item 28>

<item 29>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
You can even wear them as mourning jewelry.

This Pathfinder 2nd edition-compatible setting has been as enchantingly addictive as the fictional ichor that propelled Ourichor into the renaissance era that
the project is named after. While there are only two exquisitely illustrated supplements on creatures and equipment available currently, I recommend those with
a love of the unusual to keep an eye out for more in the years to come, such as the upcoming Paragons of Luminant Age that introduces NPCs and explores their
motives through the four moons.

Marvel Multiverse Role-playing Game

As a role-player, I’ve spent a lot of time looking for a good lightweight superhero RPG. I’ve tried quite a few but had yet to find one that really hit the mark
for me until I tried Marvel Multiverse Role-playing Game. I have to say, I was initially dubious about this game but it has blown me out of the water.
</item 29>

<item 30>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
(The Justice Department ultimately concluded its investigation without charging Gaetz.)

Looks like Matt Gaetz foray into streaming on Twitch is working out pic.twitter.com/S4GEDbgu8v — The Serfs (@theserfstv) September 30, 2022

Platform culture matters

To engage with users online, politicians must understand the culture of the platform. The streamers who appeared on Ocasio-Cortez’s stream brought their own
fans, who may not have been politically engaged at all before their favorite creator introduced it to them. As a regular gamer herself, the representative is
also up to date on the most popular games on Twitch; in 2020, it was “Among Us,” but in her most recent stream, she played the cooperative puzzle game “Pico
Park.” Gaetz, on the other hand, fumbled his launch by treating his stream like a podcast recording, limiting the back-and-forth banter with viewers that makes
Twitch so engaging.
</item 30>

<item 31>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
It is up to you to go on a perilous quest of your own to retrieve your master’s fabled weapon in time for their final duel.

Throughout this transformational quest, you use three six-sided dice to resolve mechanics for racing against time, overcoming treacherous encounters, and
uncovering new truths about your master’s shrouded past. Embrace your inner doggy, buy this 17-page game, and begin your adventure today! Swiftly now — your
master’s fate is in your paws!

Home

Wesley “Nox” Crowe, aka Noxweiler Berf, creative director for Hunters Entertainment

Doug Levandowski and Yeonsoo Julian Kim’s Home drew me in immediately, and has been one of my most recommended tabletop role-playing games. If you know me or my
work, then the fact that it’s a horror game probably won’t come as much of a surprise. It’s a haunted house exploration simulator that can be played with a full
table of players or alone as a solo journaling experience.
</item 31>

<item 32>
Cell phone shocker as 97% of kids use their device during school hours and beyond, says study (FOX News - Health, 2023-10-01T09:05:26+00:00)
Licensed clinical social worker Jennifer Kelman in Boca Raton, Florida — who provides expert advice on the Justanswer platform and was not involved in the study
— shared some of the negative repercussions of kids spending large amounts of time on their phones.

"They lose the ability to have face-to-face interactions, and to converse and connect with others," she said in an interview with Fox News Digital.

"The phones and all the apps aren’t real-life interactions, and there is beauty in connecting with others face-to-face that is lost," Kelman went on.

Kids also lose out on developing problem-solving skills, since they aren't typically tasked with negotiating "tough spaces" or navigating human challenges, the
expert added.

BACK TO SCHOOL: TIPS FOR HOMESCHOOLING THE KIDS AS MORE PARENTS DITCH CLASSROOMS

"The more apps we use, the more likely it is that we lose parts of ourselves, our confidence and self-esteem," said Kelman.
</item 32>

<item 33>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
You play as a broody Clive Rosfield, a young man whose life’s work is to protect his little brother, Joshua. The story begins when Clive’s life takes a turn for
the worse and he vows to destroy the monster who ruined his and his family’s legacy.

Developed by Creative Business Unit III, Square Enix’s internal team behind the MMORPG Final Fantasy 14, 16 leans into patchwork territories of fantasy genre
fare. There is palace intrigue, a whole lot of sex, and endless war between nations. But the developers then sprinkle in Final Fantasy elements like mother
crystals, dazzling kaiju fights between summons (known as Eikons in this iteration), and of course, Chocobos.

The quality of the story in this long and linear character-driven RPG waxes and wanes, but the action combat is among the best I’ve ever played. The gameplay
grips you from the very beginning as Clive smoothly dashes, parries, and swings his giant sword and varied magic with a dazzling amount of style.
</item 33>

<item 34>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The book itself has a stunning and distinct art style that immediately transports you into the doom metal fantasy of the setting; but that’s only a small part
of its allure.

Unlike many TTRPGs that one merely has aspirations of playing one day, Mörk Borg does everything in its power to make the game easy to run. Although it’s a
short read, no line is wasted. Even item descriptions are dripping with hints at nefarious deeds one must have done to acquire them. The optional classes are an
absolute delight for those of us that savor the idea of playing the unsavory. From the esoteric introverts that choose to make their hovel far from prying eyes
to the unfortunate discarded-at-birth cutthroats who’ve had to scratch out a living any way they can, there is something for everyone. There is always the
option of creating something more tailor-made if none of the classes are appealing, but I appreciate that Mörk Borg offers beginner-friendly ideas and
archetypes to help you create something that fits right into the gritty setting.
</item 34>

<item 35>
TwitchCon 2023 was better than last year, but everyone hated Las Vegas (TechCrunch, 2023-10-26T00:15:43+00:00)
Las Vegas’ frenetic pacing is enthralling for many, but some Twitch users complained that the city’s energy didn’t fit the Twitch demographic. Moni’s UserVoice
post noted that the sensory overload that makes Las Vegas feel like Las Vegas isn’t as alluring for introverted gamers.

“I want to be mindful of the target audience Twitch serves — I would probably assume many of us felt the same about the overstimulation,” Twitch streamer
herokerrey commented. “While I understand in-person events are a hype/stimulating experience, pairing it with one of the most active cities in the world may not
have been the best pick.”

Other users commented that they missed the spontaneous get-togethers that made previous conventions so enjoyable, and that the offsite events like creator
mixers were scattered across the city. One commenter complained that transportation between events took as long as actually attending the events.
</item 35>

<item 36>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
And there is already loads of content to support that kind of time investment. —RG

Related Diablo 4 activates and frustrates my lizard brain

27. Hitman World of Assassination: Freelancer mode

Developer: IO Interactive

Where to play: PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

Hitman World of Assassination’s Freelancer mode, which debuted in January and puts a roguelike twist on Agent 47’s globe-trotting murder-for-hire missions,
isn’t for the faint of heart. It demands constant improvisation, to a degree that can challenge even veteran Hitman players. Repeated failures can make it feel
more frustrating than fun.

But as every fan of roguelikes knows, these high-stakes experiences have the capacity to deliver a sense of exhilaration like nothing else. Pulling off multiple
daring kills, hiding the evidence (or going out with a bang), and trying to make it out alive with all the gear you brought into the mission — it’s tense and
thrilling, every time.
</item 36>

<item 37>
Discord’s latest teen safety blitz starts with content filters and automated warnings (Engadget, 2023-10-19T15:00:31+00:00)
In June, NBC News identified 35 cases over a period of six years where adults had allegedly used the platform to kidnap, groom or sexually assault minors. The
outlet also found 165 cases where authorities were prosecuting adults for sharing CSAM over Discord or allegedly using the platform to extort young users into
sending sexual images of themselves.

In the aftermath of the report, Discord banned teen dating servers and changed its safety policies to ban the sharing of AI-generated CSAM. It also announced
new rules explicitly prohibiting any other kind of text or media content that sexualizes children. At the same time, the company introduced Family Center, a
dashboard for parents and guardians to keep track of what their teens are doing on Discord.

Discord

“Teens are still in the process of learning about themselves and what it means to be online, and our new safety features are rooted in providing them with
agency on the platform, to give them ways to learn and grow in safe and age-appropriate online spaces,” said Savannah Badalich, senior director of policy at
Discord.
</item 37>

<item 38>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Players take on the role of soldiers and other hangers-on just trying to make it out alive. The heady mix of dice-driven gunplay and a playing-card-style
encounter creation makes it easy to run. But the fun begins even before that with character generation, which uses old school randomized tables to simulate an
entire military career. If you’re looking to get your video gaming buddies around the table for something that doesn’t require a headset or shouting down
pre-teens in a glitchy public lobby, the all-inclusive starter set comes highly recommended.

Project ECCO

It’s a near-impossible task to get a consistent group together, so solo TTRPGs have become my primary method of gaming. However, the barrier for me is finding a
game with a balance of narrative and mechanics that isn’t overwhelmingly crunchy, but also isn’t open-ended to the point of choice paralysis. For me, Project
ECCO found that balance.
</item 38>

<item 39>
VTubers have reached their inevitable conclusion (Polygon, 2023-10-19T16:01:50+00:00)
“Let people fail and still have that opportunity for growth in the future. More risk-taking and trying to branch out in other ways instead of simply just
streaming or following preestablished methods.”

According to both of the people we interviewed for this story, this is an agreed-upon sentiment — that to prevent further stagnation, this not-so-niche form of
content creation needs to reinvent itself and allow more freedom to its talents. Time will tell if larger agencies like Hololive (which reported 50% growth in
its 2023 financial report) and Nijisanji will allow their talents to break the mold.
</item 39>

<item 40>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
All that, and it fits in my pocket.

What Dust Remains

Danny Quach, Digital Thiccness, TTRPG writer, designer, and performer

Despite my generally sunny disposition, big smile, and boisterous laughter (as well as being deemed the internet’s emotional support himbo), I love all of my
emotions — the good, bad, and ugly. I love emo/screamo/post-hardcore music and exploring the darkness through lyrics. I love the gothic and macabre, finding
comfort in losing something you once had. I love horror movies and the inherent queerness of being an outcast being pushed to their limits. And I love games
that explore themes of melancholy and loss. Enter: What Dust Remains by 2022 Diana Jones Emerging Designer award winner Bianca Canoza — aka Momatoes.
</item 40>

<item 41>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The gameplay didn’t just help me stick with the game, but instead allowed my excitement to bubble over every time I took on a new mission. —Ana Diaz

15. Cyberpunk 2077: Phantom Liberty

Developer: CD Projekt Red

Where to play: PlayStation 5, Windows PC, and Xbox Series X

Like the best DLC, Phantom Liberty does as much to expand on the base game as it does to reframe it. Alongside the sweeping Patch 2.0, which revamped Cyberpunk
2077’s role-playing progression systems, improved its enemy AI, and course-corrected a litany of other details, Phantom Liberty also adds a whole new district
to the dystopian world of Night City, complete with its own spy-thriller storyline.

The resulting Cyberpunk 2077 is not an entirely different beast than the one that was released in 2020, but it is a more evolved one.
</item 41>

<item 42>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Playing Exquisite Biome feels a little bit like being David Attenborough on an alien planet.

With beautiful, evocative art and short, easy-to-parse rules, Exquisite Biome is my go-to “what if we made up some weird little guys and set them in motion for
an hour” game. And truly, what higher calling is there?

Fetch My Blade

Did you know you can play one of the best tabletop RPGs whenever you want, without any scheduling nightmares or unfeasible time commitments? You can when you
play Fetch My Blade, a solo journaling role-playing game created by Kelly Tran and Ethan Yen!

In Fetch My Blade, you role-play as a friendly or fierce dog of a retired legendary sword master. Create your character by choosing one of six dog breeds,
including shih tzu, corgi, mastiff, border collie, husky, and chihuahua, or select your own! Determine three truths about your master and aid them when a
mysterious stranger challenges them to a duel.
</item 42>

<item 43>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
I also do not know how Recreate made me completely OK with picking up these cute animals and flinging them into black holes, poison clouds, or freezing tundras.
The second the match starts in Party Animals, all those cute fluffy corgis, rabbits, kitties, and ducks become my enemies. I will beat them with a bat until
they can’t wake up anymore, and I won’t think twice about it. Party Animals may have made me a monster? I don’t know.

The Gang Beasts-esque wiggly physics mixed with the cute characters makes for a perfect party game of fluffy fighting. —Julia Lee

Related September games you might have missed

45.
</item 43>

<item 44>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Influencer marketing agencies on both ends of the political spectrum are ramping up their recruitment faster than the Federal Elections Commission can regulate
the industry. This year, the conservative agency Influenceable has been recruiting Gen Z creators to rally behind far-right politicians and parrot GOP talking
points, without disclosing their pay. The tactic irked some Republicans, the Texas Tribune reported, including a Texas state representative who called for an
investigation into the company.

Given the resistance to Republican politicians in online spaces that attract young people, it’s unsurprising that candidates may rely on shadowy agencies like
Influenceable to do the work for them. Republican politicians have a reputation for botching even the most straightforward digital campaigns. In May, Florida
Gov. Ron DeSantis launched his presidential campaign on Twitter Spaces in an audio conversation with Elon Musk. The conversation started late and was so riddled
with technical issues that it ended after just 21 minutes. Critics on both the right and the left described the launch as a disaster.
</item 44>

<item 45>
The 35 best couch co-op games for Nintendo Switch (Polygon, 2023-10-20T16:20:53+00:00)
I’ve donated hours and hours of my life to this deceptively simple roguelike bullet-hell — min-maxing various buffs in order to avoid waves of mob attacks. But
multiplayer changes everything. Having more reinforcements to fight waves of attacks means each player can try out unique combinations of weapons and buffs.
Just make sure you and your teammates stay in the same screen; there isn’t split screen.

Though packing up and moving is a pain in the neck in real life, Moving Out 2 turns the experience into a goofy, whimsical group experience. Play as a member of
the F.A.R.T. moving crew (yes, you read that correctly) attempting to move furniture, and other items, around some highly inconvenient spaces.

Now more than a decade old, Diablo 3 is still a great local multiplayer choice, especially on the Switch. The action RPG, which felt bombastic when it first
came out, has become a very comforting grind game that I get to share with friends.
</item 45>

<item 46>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
It’s the first heavy-duty chamber LARP I’ve played since the pandemic; it reset my brain and helped me remember what I value in play. Players continuously swap
roles as they play through a very queer Greek myth in which young Orestes returns home and must decide whether to kill his treacherous mother and father-in-law.
Never have I planned a murder with such erotically charged practice stabbing.

Dungeon Crawl Classics

Thilo Graf of AAW Games

I run and play a ton of different RPG systems, but precious few have had such a tremendous and lasting impact on my design and games as Dungeon Crawl Classics
by Goodman Games.

With aesthetics influenced by all-time greats like Leiber and Moorcock, volatile magic, a relatively flat power curve for a d20-based game, and an emphasis on
the skill of the players, as opposed to primarily rewarding the crafting of effective builds, DCC is a joy to run and play.
</item 46>

<item 47>
CEO David Baszucki’s mission to make Roblox a billion-player platform (The Verge, 2023-10-12T14:01:00+00:00)
So we can contribute to the growth of Meta Quest, and I’m excited about that. If someday there are 500,000 or 500 million VR headsets, Roblox will be a huge
part of it, but I think we’re not in the position of predicting the growth of any hardware device.

What about AR? So not fully immersive, but AR glasses.

AR is also super interesting, right? Is the future going to be mixed reality? Is the future going to be lighter weight, less overlaid type stuff? Is the future
ultimately going to be some of the vision of some of the companies that have been started? Someday, I think I wrote a blog post about it 10 years ago, we will
have full overlay, whether it’s contact lenses, and it’ll be lightweight and all of that. I think AR is very interesting when we think about immersive 3D
communication because there’s communication where we’re simulating this world and we’re sitting here together.
</item 47>

<item 48>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
The thing we’re jacked into and collectively creating seems more like a global endocrine system than anything we might have visualized in the years while
“cyber” was a prefix. This may seem a banal observation, given that Marshall McLuhan was talking about the global nervous system more than fifty years ago. We
had enthusiasm about cybernetics and global connectivity over the decades and, more recently, a revitalization of theory about networks and kinship and rhizomes
and all the rest. (The irony is that with fifty years of talk on “systems thinking,” we still have responses to things like COVID-19 or climate that are almost
antithetical to considering interconnected systems — dominated by one set of expertise and failing to incorporate the social sciences and humanities). So —
globally connected, yet divided into silos, camps, echo-chambers, and so on. Social media platforms are acting as agents, structuring our interactions and our
spaces for dialogue and solution-building.
</item 48>

<item 49>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
But the fresh grand dame of my tabletop world is Triangle Agency, a gleaming, razor-sharp new TTRPG that feels like a cross between The X-Files, Annihilation,
and playing Control with the Robert Anton Wilson mod turned on.

In Triangle Agency, you are super-abled agents investigating and stopping ontological anomalies from destroying reality, with mechanics that feel like the
platonic ideal mix of analytic crunch and grand narrative weirdness. And my god, it is gorgeous, with jaw-dropping art and design that puts it on the same shelf
as Wanderhome and Mörk Borg. The full game ships early in 2024, but you can get the digital rules and start playing right now, for the low, low price of
“whatever you think is appropriate.”

Twilight: 2000

Charlie Hall, senior editor, tabletop

I firmly believe that tabletop role-play should always take into account the audience that sits down at the table to play. That’s part of the reason why I hate
best-of lists for this particular category of games in particular.
</item 49>

<item 50>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
Attending meetings in pajama bottoms might seem like a job perk, but some remote workers have found that videoconferencing for work isn’t all it’s cracked up to
be.

In the years since the COVID pandemic triggered a spike in telecommuting, many have complained of so-called "Zoom fatigue."

Thought to be coined by Stanford University professor Jeremy Bailenson, the term refers to a level of exhaustion that comes with interacting with others on
camera all throughout the day.

REMOTE WORKERS RESIST MONDAY OFFICE HOURS, BUT BUILDING 'ORGANIZATIONAL CULTURE' IS KEY, SAYS EXECUTIVE

Dr. Kyle Elliott, a tech career coach in San Francisco, defines Zoom fatigue as "incessant tiredness or lack of energy as a result of spending a large quantity
of time on Zoom."

Additional symptoms can include physical tension (including headaches, eye strain and stiffness), moodiness, a sense of disengagement or difficulty
concentrating, according to Dr.
</item 50>

<item 51>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This was a year of upheaval in video games. The industry has shapeshifted over the past 12 months, and it’s not all due to Microsoft’s lengthy acquisition of
Activision, Blizzard and King. While Xbox executives were defending the legality of a $69 billion deal that would create the third-largest video game studio in
the world, smaller companies were firing staff and shutting down entire teams, even amid fervent collective-bargaining efforts. It’s been a wild ride.

In 2023, the main factors molding the video game landscape were consolidation, layoffs and unionization, with each of these phenomena feeding into each other.
This past year, the video game industry shrank, even as it grew financially.

Consolidation

When Microsoft's purchase of Activision-Blizzard-King was finally approved on October 12, 2023 after nearly two years of regulatory hurdles, it became the
world’s third-largest video game studio by revenue.
</item 51>

<item 52>
TwitchCon 2023 was better than last year, but everyone hated Las Vegas (TechCrunch, 2023-10-26T00:15:43+00:00)
Twitch, which saw exponential growth during the pandemic, was one of them. Last year’s San Diego-based TwitchCon, for all of its faults, took place in a
walkable city abundant with nearby places to hang out, allowing attendees to bond, network and interact outside of the convention center. There’s no shortage of
third places in Las Vegas, but for many TwitchCon attendees, nightclubs and casinos aren’t accessible alternatives to San Diego’s more casual venues.

Moni, a streamer working in games tech who goes by Monistreams, expressed her frustrations in a recent post on Twitch’s feedback forum UserVoice.

“They do a really good job of preparing content, it truly feels like a celebration of the community and streamers,” Moni said over a Discord call. “That’s their
selling point. It is a celebration of streamers and their communities.
</item 52>

<item 53>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Galaxy-spanning faction quests give you plenty to do, but it’s the side quests that really make you feel like the captain in something Star Trek adjacent:
generation ships and superhero hideouts and colonies of clones. There’s good sci-fi here for those willing to make the journey.

It’s true that the game oversold the idea of a galaxy with over 1,000 planets, a number that pales in comparison to the game’s most obvious competitor, No Man’s
Sky. It’s a shame this became so much of the focus thanks to Bethesda’s marketing, because that same focus obscured the absolutely fascinating new game plus
mode at the game’s heart. Consider this: No Man’s Sky is the game that realizes the universe’s infinite number of planets, while Starfield realizes the
universe’s infinite number of possibilities. —Clayton Ashley

33.
</item 53>

<item 54>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Even if Jake Paul was less despised online, endorsements aren’t as valuable as they used to be. A paper published in the International Journal of Communication
this year reported that mobilizing influencers to build support for political causes is more effective than using their endorsements to gain votes. Blatant
endorsements lack personal connection, and create an “authenticity gap.” Authentic creators are trustworthy, and in campaign strategy, that’s more valuable than
the size of their following.

Martin Riedl, a University of Tennessee, Knoxville professor who studies social media and co-authored the paper, said that society tends to reward public
figures for speaking out on social justice issues because there’s more “situational awareness.”

“If you use influencers in your campaign, it’s important that they can authentically believe in what you promote,” Riedl said. “And if that’s not the case, that
makes it really difficult for people to take your campaign seriously.”

Authenticity reigns

Facebook ads, celebrity endorsements and campaign trail rallies aren’t enough to engage voters online.
</item 54>

<item 55>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
All this is wrapped in Munkao’s fantastically disgusting artwork. His vistas and cultural portraits are gorgeous, but when he brings his talents to bear on
feelers and chitin and filth, the results are revolting in the best possible way. Sadly, this is likely the final installment in the ATTI series and unlikely to
be reprinted, so get this book while you can!

Monsterhearts

Figuring out what to talk about proved trickier than I expected. A lot of the things I played this year were playtests or previews for things not out yet, which
would feel more like a humblebrag than a recommendation, or were my own games which were great to play but would feel a bit weird and self-indulgent to suggest
as the “best.”

A really pleasant TTRPG memory I had from this year was playing Monsterhearts for the first time in three years. This was one of the first games I ever played
(for ages afterwards I kept calling playbooks in every game “skins”).
</item 55>

<item 56>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
It is not as much of a heroic power fantasy as D&D 5e or Pathfinder, but neither is it as minimalist as many old-school games. With fickle gods, mutating
magics, lucky scoundrels, and ferocious fighters, DCC feels like playing through a Bal-Sagoth or Eternal Champion song. It’s also easy to pick up if your group
has any experience with D&D’s third edition or Pathfinder. DCC rocks! Its focus on simple, impactful rules was a core design inspiration for the (almost
system-neutral) Survivalist’s Guide to Spelunking for 5e, a book I wrote with Douglas Niles and Stephen Yeardley. It warms my heart to know that DCC authors,
judges, and players use the book as well!

Dungeons & Dragons

Mike Haracz, corporate R&D chef and host of Heroes’ Feast

I’m lucky enough to be in the seventh year of a homebrew Dungeons & Dragons campaign, one I play with friends locally in the Chicagoland area.
</item 56>

<item 57>
TwitchCon 2023 was better than last year, but everyone hated Las Vegas (TechCrunch, 2023-10-26T00:15:43+00:00)
One commenter complained that transportation between events took as long as actually attending the events.

DinomiteTwins, Twitch streamers who are identical twins and share a channel, commented that the city’s size “put a damper” on how easily they could meet fellow
creators.

“We were in an approved hotel [provided by Twitch] and I met more vacationers, metal workers and brides than Twitch streamers in my hotel,” one of the
DinomiteTwins commented. “A smaller walkable city promotes organic interaction as we are not all taking cabs … It felt like the priority was just being in a fun
place instead of facilitating vendor, creator and community meetups and deepening those ties.”

Third places provide more than just a break from the glare of fluorescent convention center lighting — they’re vital for building community, a sentiment that
Twitch insists is unique to its platform as other livestreaming competitors gain users. Compared to its competitors, Twitch’s robust moderation and slew of
collaboration features does facilitate a closer knit sense of community among its streamers and viewers.
</item 57>

<item 58>
TwitchCon 2023 was better than last year, but everyone hated Las Vegas (TechCrunch, 2023-10-26T00:15:43+00:00)
There was plenty of opportunity to mingle within the walls of the convention center, which even had a designated room to decompress and make friendship
bracelets, but organic interactions seemed constrained to Twitch-sponsored events. The diverse programming and various meetups could only do so much in an area
with so few casual social spaces.

Lowco, Moni and multiple other Twitch users commenting on Moni’s UserVoice post all questioned why TwitchCon couldn’t take place in San Diego again. Some have
floated hosting both a West Coast and East Coast TwitchCon, if capacity is a concern. VidCon, which has traditionally been held in Anaheim, California, recently
launched a second convention earlier this year in Baltimore, Maryland.

“Ultimately, TwitchCon is a community celebration,” Lowco said. “We are there to meet up with our viewers and fellow streamers. Let’s pick places that make it
easy for us to gather.”
</item 58>

<item 59>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
I can’t recommend it enough. Grab a copy of the PDF, turn down the lights, and start mapping your own haunted house!

The Luminant Age

I became an immediate fan of Luminant Age after being invited to be a contributor. What caught my eye at the start was the Four Humanities ancestries named
Blood, Clay, Silk, and Horn. This year, many of my Thursday mornings have been a whirlwind of weird as our playtest party encounter teleporting flickrats, heal
via esseweaving, and get hunted down by cultists in stryx bonemasques. The city we’re in, Ourichor, is built on giant oil-rig-like platforms in the open ocean.
There are no sun nor stars, but three moons... that you’re allowed to see.

Also, did I mention the predominant religion featured, the Angelites, memorialize their dead by preserving their eyeballs? You can even wear them as mourning
jewelry.
</item 59>

<item 60>
Pokémon is no longer just a game — it’s a lifestyle (The Verge, 2023-11-23T18:00:00+00:00)
More than 25 years since the original game’s release, multiple generations of kids have become adults under the watchful eyes of Pikachu and friends, and the
series has grown to encompass everything from anime and merchandise to card games and Hollywood blockbusters. Perhaps most surprisingly of all, with the growth
in popularity of trading card games like Pokémon in a postpandemic world and the record-breaking sales of the recent generation of titles despite quality
concerns, it only seems to get more popular by the year.

For a franchise to retain relevance for as long as Pokémon, it must evolve — and not just evolve but interweave its ideals into the fabric of daily life. Those
early generations of kids need to pass on their love of Pokémon like how Disney animated classics often serve as a formative childhood introduction to film.

Perhaps nothing encapsulates Pokémon’s successful transformation from video game to a way of life quite like PokéTsume.

PokéTsume. Image: TV Tokyo

But what even is PokéTsume?
</item 60>

<item 61>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Taught by the designer herself, I got to play this at Big Bad Con 2023 with two other strangers who quickly became co-collaborators in this tragic world we
built in under two hours that told a story of a legacy that wasn’t quite within reach. Impassioned and intriguing card-prompted questions along with focused,
intense moments of dice rolls that helped move the story along, What Dust Remains challenged me to discover, explore, and sit in moments of despair and
desperation and had the whole table contemplating what parts of us are we going to sacrifice to leave a legacy that would be bigger than we could ever be
behind.

The Wildsea

Earlier this year, I searched for the perfect system to kickstart my new actual-play campaign with three “must-haves” in mind: a vivid world, rich character
options, and simple but punchy rules. Through sheer accident (providence?), I stumbled upon The Wildsea by Felix Isaacs and Mythworks — and it was love at first
sight.
</item 61>

<item 62>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
It won’t always be right, though. But it becomes the epicenter of where the conversation is. And then I’ll think what percentage of our population really wants
to earnestly take the breath and say, let me stop doomscrolling, and let me just wait until people can do the work, synthesize it and then help me react?” he
continued.

“We are also so plugged into all these storylines, whether it’s from traditional media, whether it’s from user-generated — someone on their phone, you know,
posting to a Telegram group. We are overwhelmed and overloaded, and it’s unfortunately appealing to the worst parts of a lot of our instincts,” Ohanian added.

In part, some of the issues around how social media platforms developed had to do with how the companies were originally developed. Ohanian, for example, shared
that Reddit grew out of his running of a forum site in college, when he believed that online community felt as real as offline.
</item 62>

<item 63>
Fortnite is the latest game to entice players with a portal to the past (TechCrunch, 2023-10-30T19:18:39+00:00)
Elizabeth Harper, editorial director of the longtime dedicated World of Warcraft news site Blizzard Watch, told TechCrunch that Blizzard actually reluctantly
launched the old-school version of its game in response to demands from its own community.

“Before WoW Classic originally launched, third parties were already running their own versions of ‘classic’ servers,” Harper said. “Game mechanics and class
balance were extremely different then, and there was a lot of nostalgia for the community that existed back in the day.”

While Fortnite players, like WoW players, have a ton of nostalgia for the older version of the game, its maps were comparably simple and difficult to traverse.
For players accustomed to gaming’s modern quality of life improvements, old games can feel sparse and clunky if not reimagined outright.

“WoW Classic has been a runaway success, and fans can’t get enough of re-releases of classic content, devouring each expansion and raid tier as they’re rolled
out.
</item 63>

<item 64>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
"And this meant a lot of time looking at each other on-screen instead of in person."

Long after the pandemic, many of these habits have persisted.

"This video-ing of each other, it turns out, is far more tiring to humans than actually being together," said Nassar.

"One reason is that our brains are wired for actual in-person interactions, not wired for interactions on the screen."

Impacts of Zoom fatigue

While video calls can be convenient, Elliott warned that staring at a computer or phone screen for extended periods of time can result in negative mental health
consequences .

"Zoom fatigue can aggravate stress, anxiety and burnout among workers," he told Fox News Digital.

"Our brains are wired for actual in-person interactions, not wired for interactions on the screen."

"Some people share that spending so much time interacting on screens makes it difficult to interact in real life."

CAN YOU AVOID END-OF-THE-YEAR CAREER BURNOUT?
</item 64>

<item 65>
VTubers have reached their inevitable conclusion (Polygon, 2023-10-19T16:01:50+00:00)
Brands, too: Remember the Tony the Tiger VTuber in August of 2022?

Many creators that jumped on the trend have since stopped using their avatars, but at the start of the COVID-19 pandemic, there was a large increase in VTubers
on Twitch and other streaming platforms, like YouTube, as models and equipment became more accessible to the general public. Twitch has since gone on to
highlight a wide variety of VTubers through its VTuber Takeover promotion, which most recently ran in August. Twitch also reported that VTubing-related content
on the platform shot up by some 350% between January and August 2022. Despite this, some have argued that the VTubing bubble has all but burst.

Polygon is celebrating TwitchCon 2023 with a package of stories covering streaming trends and news from the show. Check it out!

“I do feel that VTubing has become less novel as the market has become more saturated,” says a streamer with agency Nijisanji who is not approved to speak
publicly.
</item 65>

<item 66>
With Amo, the founder of Zenly wants to make social apps social again (TechCrunch, 2023-11-19T12:07:20+00:00)
“The [World Health Organization] now calls it the loneliness epidemic. And if they say that it’s an epidemic, it’s because it’s actually infectious. In other
words, if you’re isolated, your loved ones are too because you’re unreachable. So during the two hours you’re on TikTok, they have no one to talk to,” Martin
told me.

“And at the same time, the human needs that the social consumer space can fulfill are no longer covered by these products, whereas they used to be,” he added.
“In the early days of Facebook, I don’t know if you remember, profiles were sort of a mash-up. There were drawings, games, photos, text. You’d write long
comments, it could be a poem . . . And on the other hand, it was a reminder that you mattered to these people.”

According to him, the current generation of social networks are very passive.
</item 66>

<item 67>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
), got every player invested in our characters and each other (even in a group of total strangers), and allowed the whole table to lean into both the strange
and ridiculous nature of life. This year, it was just the kind of joy that I needed.

2023 was a personal tabletop gold mine. We finally got Brennan Lee Mulligan’s first official 5e homebrew; I discovered tabletop meetup hub and game master meat
market StartPlaying.Games, where you can, as I have, hire a handsome Scottish man to D&D dominate you and your friends every week; and I played a ton of my new
favorite party game, As You ’Wich. It’s a dead-simple hand-builder that, for me at least, boils down to convincing your nephews to eat disgusting sandwiches
(hair, mustard, and broken glass on marble rye, absolutely not toasted).
</item 67>

<item 68>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Our campaign is called The Heartsguard Saga, and it’s based off Deven Rue’s map of a place called Euphoros. My players are at around level 15 and I frequently
give them a variety of amazing and unique gear that I’ve dreamed up all on my own... which they go on to forget they have, and never use. We used to play in
person, but since the pandemic I run the game virtually online from my house. I put a camera hovering over my gaming mat, and make a labeled grid for the
players so they call out, “Move me from A5 to B10.” During important rolls, I even make my players face their camera toward their dice for added suspense!

Petrana Radulovic, entertainment reporter

I’d wanted to play Dungeons & Dragons since I was in high school, but I never knew how to get into it.
</item 68>

<item 69>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
It was all live-service games like League of Legends and Fortnite.

In 2011, when Twitch first launched, content creation wasn’t fully a career and people still uploaded their photos to Facebook. Normal, everyday teens weren’t
yet decrying the hungry maw of the internet, and the generational shift toward work-life balance hadn’t yet taken hold. Twitch happened at the exact right time,
benefiting from a cadre of optimistic young users who grew up on the internet, happy to give away as much of themselves as they could. They didn’t even realize
it was happening, in many cases, and they couldn’t imagine the unsustainable system it would go on to create. People were just having fun.

Things have changed since then. Twitch has crowned millionaires, like Félix “xQc” Lengyel, who wear thick diamond chains around their necks while streaming. At
the same time, viewership is dropping after years of platform growth that was partially bolstered by the COVID-19 pandemic, all while top creators are streaming
less.
</item 69>

<item 70>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
“The amount of media and content that we consume, political and cultural, where people get their information … It’s getting more and more intertwined,” Henry
said. “To be effective ideologically, when it comes to campaigns and these wins that we want, we have to understand how these ecosystems work, and be
strategically using them or else we risk being left behind.”

Engaging with voters online requires more than just working with popular creators. In Ramaswamy’s case, an endorsement from Jake Paul only drove animosity from
TikTok viewers. While his older brother, Logan Paul, has made the occasional social commentary on his podcast, the younger Paul has rarely engaged with politics
or social issues. Jake Paul is generally disliked online for the running list of allegations against him, including sexual assault accusations.

The Paul brothers have a reputation of promoting anyone and anything for a paycheck, and when Jake Paul posted a political endorsement out of the blue, viewers
immediately questioned how much Ramaswamy paid him.
</item 70>

<item 71>
TwitchCon 2023 was better than last year, but everyone hated Las Vegas (TechCrunch, 2023-10-26T00:15:43+00:00)
“It was a huge demand to come back. I think fandoms have grown in that time period, just the fervor around specific creators that was totally and wholly
different than we’d seen before,” Delphin said of last year’s TwitchCon. “And the creators themselves had also absolutely exploded. It used to be that you’d
come to TwitchCon, be the biggest creator, and you could walk around pretty freely. Now it’s huge, it’s a whole different kind of apparatus to manage.”

Attendee feedback is clear: Next year’s convention, wherever it is, needs to have more accessible third places.

The third place is a sociological term that refers to a physical space that isn’t work or home, where people can socialize without the obligations of
productivity. If you can hang out there, it’s a third place. During the pandemic, virtual communities served as ad hoc third places in lieu of coffee shops,
bookstores and gyms. Twitch, which saw exponential growth during the pandemic, was one of them.
</item 71>

<item 72>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Much like humanity itself? —ML

23. Paranormasight: The Seven Mysteries of Honjo

Developer: Square Enix

Where to play: Android, iOS, Nintendo Switch, and Windows PC

Square Enix has plenty of mega-franchises to fill its time (and its coffers). This year, we have new entries for Octopath Traveler and Final Fantasy, along with
new Dragon Quest and Kingdom Hearts games in the not-so-distant future. Dayenu!

And yet, the publisher can’t help itself from bombarding us with surprising, interesting, sometimes great, often good-enough experiments. In 2022, we got an
English-language remake of lost gem Live A Live, the surprisingly enjoyable tactical RPG DioField Chronicle, a bonkers Final Fantasy spinoff featuring the
musical stylings of Limp Bizkit, and a pair of oddball card games lathered in lore from gaming’s best weirdo.
</item 72>

<item 73>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It’s a time in your life where you feel so, so much. You can see that earnestness in the teenage experience where everything is a big, huge issue — sometimes to
the point of cringe — tied up in that big, global issue of the meteor that’s looking to destroy everything. —N. Carpenter

Related August games you might have missed

40. World of Warcraft Classic

Developer: Blizzard Entertainment

Where to play: Mac and Windows PC

Something very, very interesting started to happen in the retro version of WoW this year. While Blizzard has been content to march one half of the game’s
community forward through its history of expansions, it’s started to come up with creative ways to keep the other half — the half that wants to stay in the game
as it was at launch — engaged.

The first of these was Hardcore, a brilliant permadeath mode that instantly made this aging game both more dangerous and more social, resurrecting the spirit of
its 2004 servers.
</item 73>

<item 74>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The Quiet Year

Em Friedman, associate professor of English at Auburn University and Polygon contributor

Look, The Quiet Year appeared not once but twice on last year’s roundup, I know. Jay Dragon praised its elegance and Keerthi Sridharan noted the way their game
group used it as a prologue (as Friends at the Table and The Adventure Zone have popularized), and it’s a standard early on in my TTPRG class. But for me it was
the standout game of this year because after running it for my niece (then 4) and nephew (8), my nephew asked that I run “the skulls game” for his entire class.
While not intended for a dozen grade schoolers, it turns out the mechanics of The Quiet Year — which ensure everyone gets to have their turn unimpeded by the
commentary of others, ritualizes dissent and discussion, as well as having a glorious sheet of blank paper to draw the zombie chicken monster friend of your
dreams — turned out to be a perfect framework to imagine how to work together.
</item 74>

<item 75>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
"All these video-based calls can quickly lead to video fatigue if you're not mindful and take care to protect your mental health and well-being ."

Strategies that can help

Just because a meeting is scheduled on Zoom or a similar video conferencing platform doesn't mean you need to be on video for the call, Elliott pointed out.

"When it makes sense, particularly if it's a large meeting where your input will be less frequent or nonexistent, you might opt to turn your video off from time
to reduce Zoom fatigue," he advised.

The expert also recommended taking adequate breaks away from the computer, even if it's for just a few minutes, throughout the day.

"It can be tempting, but try to avoid looking at another screen, such as your phone, during this break, since you want to avoid swapping one digital toxin for
another," Elliott said.

For those leading a video meeting, Corcoran suggested starting with a quick off-camera stretch break or mindfulness exercise to increase attention and focus.
</item 75>

<item 76>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Ray reassured me that I was not crazy. Google results today do feel different from how they felt just five or six years ago for two major reasons. The first was
Google’s response to the disinformation panic around the 2016 election, which involved questioning the notion that the most reliable information could be chosen
by a form of popularity, meaning how many links a site received from other sites. As a result, the algorithm seemed to change its approach to links, especially
when it came to news and sites offering legal, financial, or health advice, and instead paid more attention to what Google came to call E-E-A-T: experience,
expertise, authoritativeness, and trustworthiness.

“E-E-A-T has had a pretty big impact on what types of results you see,” Ray told me. She’s done extensive (and fascinating) research around how certain sites
have fared under these new guidelines: Urban Dictionary, down! Mayo Clinic, up! Some people consider EEAT part of what’s making results better than ever.
</item 76>

<item 77>
Pokémon is no longer just a game — it’s a lifestyle (The Verge, 2023-11-23T18:00:00+00:00)
Over 5 million unique users accessed Yahoo! Kids every month at this time, all within the series’ core target demographic. In the end, the project proved
successful, with Diamond and Pearl selling one-third more copies than its predecessors Ruby and Sapphire before eventually eclipsing the lifetime sales of those
titles.

Notably, at the time, it was one of the company’s biggest attempts at diversifying the brand since its initial launch. Following the initial multimedia blitz,
and the introduction of a World Championships and competitive circuit for the video games and TCG, the series remained relatively static. At this point, Pokémon
was yet to truly experiment with online distribution and promotion, nor had attempts been made to branch out beyond the series’ core audience and multimedia
properties following the initial surge in popularity.

As such, Pokémon Garden was the first real attempt at expanding the core experience of Pokémon to other platforms and audiences.
</item 77>

<item 78>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
At its best, you really can feel the Force around you. —CA

31. Connections

Developer: The New York Times

Where to play: Android, browser, and iOS

After the acquisition of viral hit Wordle, The New York Times’ Crossword ecosystem leveled up to full-fledged attention competitor — NYT is a media company with
a gaming platform. And its staff of puzzle writers and editors are not stooping to mind-numbing mobile content to keep the expansion going. Connections, its
latest title to pair well with morning coffee, is one of the year’s best games.

Connections offers you a grid of 16 words and a mission: Detect the common threads between four different sets of words without embarrassing yourself. The
grouping logic ranges from simple (“animals”) to silly (“synonyms for farting”) to sneaky (“countries when the letter ‘A’ is added”).
</item 78>

<item 79>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
Julia Corcoran, a licensed clinical psychologist and board-certified coach, who is currently serving as the director of clinical strategy and experience at
Modern Health in San Francisco.

The challenge isn’t exclusive to Zoom; other videoconferencing platforms can have the same effect.

The pandemic's effect

With the COVID-19 pandemic, Zoom and other videoconferencing platforms "exploded in popularity," said Dr. Carl Nassar, PhD, a professional counselor based in
Denver, Colorado.

"We all found ourselves at home looking at the people we worked with, the people we went out for drinks with and the people in our extended family on a computer
screen," he told Fox News Digital.

WORKERS FIRED ON ZOOM: HERE'S HOW TO NAVIGATE A 'DEVASTATING' EXPERIENCE AND COME OUT ON THE OTHER SIDE

"Our work went virtual, our social life went virtual, our doctors' appointments went virtual, and even our intimate relationships sometimes went virtual," he
said.
</item 79>

<item 80>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
Or connections will be pretty bad because we can’t mobilize players together in a single match. There’s all these cascading effects.”

While a Twitch streamer’s livelihood depends on having a steady stream of content, the system taxes them, too. Tenney’s goodbye to his fans was a crushing one,
because the path that led him there was fueled by lost youth and a dream. Over time, the promise of a dream job — get paid to play games! — became corporatized,
and contracts turned things ugly. Tenney said he started feeling trapped, and he didn’t like how much his decisions started being motivated by money. Tenney is
hardly alone here now that Twitch has “grown up.”

“There is inherently [a] vicious and negative cycle for everyone involved, for how live service interacts with Twitch as a medium,” the game developer, who also
spent years as a serious Twitch streamer, says. It’s been years, but he says he’s still recovering from the burnout of livestreaming all the time.
</item 80>

<item 81>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Media theorist Geert Lovink reflects on a 1996 interview with John Perry Barlow, Electronic Frontier Foundation cofounder and Grateful Dead lyricist, in which
Barlow was describing how cyberspace was connecting each and every synapse of all citizens on the planet. As Lovink writes, “Apart from the so-called last
billion we’re there now. This is what we can all agree on. The corona crisis is the first Event in World History where the internet doesn’t merely play ‘a role’
— the Event coincides with the Net. There’s a deep irony to this. The virus and the network ... sigh, that’s an old trope, right?” Indeed, read through one
cultural history, it seems obvious that we would reach this point of being globally networked, and that the Internet would not just “play a role” in global
events like COVID-19 or climate change, but shape them.
</item 81>

<item 82>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
It feels forbidden somehow, even as it’s obviously harmless.

But Dixon also captures something ephemeral and nostalgic about what it’s like to be in a niche online community and group chat, and the magic of making
discoveries in those spaces. On an increasingly centralized internet, that experience has become rare, and I miss it.

Exquisite Biome

Everest Pipkin, game and software developer

I’ve heard Exquisite Biome described as “Spore for the kitchen table,” and if that isn’t a great pitch I don’t know what is.

Coming from Caro Asercion (game designer) and Si Sweetman (illustrator), Exquisite Biome is in the world-building TTRPG tradition, but it focuses specifically
on ecological biomes and the creatures that populate them. Through short generative prompts tied to a deck of playing cards, you build and witness the lives of
animals far beyond the edge of our own world. Playing Exquisite Biome feels a little bit like being David Attenborough on an alien planet.
</item 82>

<item 83>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
On Twitch, debate streams between creators across the political spectrum are wildly popular, and between May 2021 and May 2022, the site’s Politics tag tripled
in viewership. The platform itself is a haven for some figures on the far-right who have since been kicked off of YouTube.

Republican politicians, however, have failed to establish the rapport with young voters that their Democratic counterparts have. While X, the site previously
known as Twitter, has become a conservative incubator, Republican politicians haven’t taken advantage of the platform the way far-right influencers have.

That gap in digital fluency is most apparent in the way politicians approach new platforms. In 2020, Ocasio-Cortez encouraged viewers to vote in the upcoming
presidential election by inviting the most popular streamers to play the pandemic’s most popular game on her newly launched Twitch channel.
</item 83>

<item 84>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Scholars have argued that putting too much emphasis on the platforms can be too simplistic and reeks of technological determinism; they have also pointed out
that cultures like the United States’ and the legacy media have a long history with post-truth. That said, there are certainly dynamics going on that we did not
anticipate, and we don’t seem quite sure what to do with them, even with multiple areas of scholarship in communication, disinformation, and social media and
democracy working on these inquiries for years.

What seems clear is that the Internet is not the connectedness we imagined. The ecology and spirituality of the 1960s, which shaped and structured much of what
we see as energy democracy and the good future today, told us we were all connected. Globally networked — it sounds familiar, like a fevered dream from the
1980s or 1990s, a dream that in turn had its roots in the 1960s and before.
</item 84>

<item 85>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
It uses custom tarot cards and a whiteboard to track your character’s progress as they dare to shine a light on the chilling danger of a variety of potential
scenarios full of ghosts and other dark entities.

​The action is propelled through the use of the custom tarot cards — the Night Deck — which are filled with prompts that require the creative interaction of
players. Each round, players take turn as the director of the story and guide the narrative forward using the guidance of the cards to describe and draw each
room of the haunted house that they are exploring. The game encourages collaboration in determining the story’s direction and does a remarkable job of steering
the action forward without a single game master putting forth the often large amount of preparation that role-playing games usually require.

​As the game progresses, characters collect wounds and push past the terrors that await them and work together to resolve the mysteries at the root of each
horror. The solo mode of the game plays just as smoothly as the group experience, leaning on the prompts and player’s imagination to create a journal
documenting the character’s harrowing encounter.
</item 85>

<item 86>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

"Whenever you’re able, suggest audio-only meetings that don’t require screens so people can do things they would naturally do in an in-person setting, like move
around or look out a window," she advised.

"Walking away from our screens can reduce the urge to multitask, reduce eye fatigue and increase our capacity to move around."

Turning off the self-view option can also be helpful, Corcoran said — "it can be particularly distracting and draining because we’re not used to seeing our own
images so much."

As the use of video platforms is still relatively new for many professionals, Corcoran noted that the long-term effects aren’t yet known.

CLICK HERE TO GET THE FOX NEWS APP

"That being said, we know that constant feelings of fatigue, exhaustion and physical strain or tension do correlate with negative mental health outcomes," she
said.

"It’s important to take steps to reduce Zoom fatigue before it leads to things like chronic stress or burnout."
</item 86>

<item 87>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Furthermore, in the process of putting a game together, I found a glut of free online tools from dungeon generators to NPC character sheet generators that made
game prep a breeze. I find myself coming back to this game again and again with each game session only giving me ideas for future campaigns. I know I’m not
alone in this feeling given the number of Mörk Borg-compatible games that have sprung up this year, from Pirate Borg to Chris Lockey’s upcoming Abyss of
Hallucinations.

I’m Sorry Did You Say Street Magic

Madison Durham, writer and journalist, staff writer at Reviewed

To me, some of the most joyful moments in tabletop gaming come from the experience of creative collaboration — the magic of making something new with friends,
be it a narrative, a character, a world, or in this case, a city. I’m Sorry Did You Say Street Magic by Caro Asercion is a GM-less city-building game, played
between a group of two to six people with a stack of index cards and your favorite writing utensils.
</item 87>

<item 88>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
What difference does it make if something scores highly on some arbitrary rubric if it doesn’t resonate with the people in your community? And so I look at
year-end lists like these as a kind of menu, boards of fare that clever game masters can use to pick just the right flavor for a given table.

For me this year was all about reconnecting with the other dads in my neighborhood after years spent locked inside during the pandemic. I spun up a Discord that
we used to organize friendly games of Call of Duty’s DMZ mode, organized a few visits to the local brewery to catch up make new connections, and generally just
tried to get awkward, middle-aged men accustomed to seeing each other socially again. One night I decided to bust out Twilight: 2000, and it was an instant hit.

Twilight: 2000 is a game about surviving in the aftermath of a fictional nuclear conflict in the heart of Europe. It’s an alternate history where the Cold War
went hot, swallowing up an entire United States Marine division.
</item 88>

<item 89>
CEO David Baszucki’s mission to make Roblox a billion-player platform (The Verge, 2023-10-12T14:01:00+00:00)
And I think our vision of what we’re working on goes really almost to the future of communication. It goes to the evolution of the mail system, to the telegraph
system, to the phone system, to, as we saw in covid, the video system. We just think inevitably there’s a generation beyond that, which is immersive 3D
communication, whether it’s playing together, whether it’s trying to graduate from high school together during the midst of covid, whether it’s a simulation of
our Roblox office inside Roblox. We go and have a simulation of our office, and we come together for serendipitous events, whether it’s going to a concert. So
our belief is, this type of technology is bigger than gaming. Gaming is a part of it, side-by-side concerts and working together, so I think we’ve evolved to a
utility vision of this type of platform.

I think you made a prediction at RDC a couple of weeks ago about dating.
</item 89>

<item 90>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Here’s what we found.

City of Winter

Linda Codega, Ennie Award-winning entertainment journalist

My friends and I still talk about the weekend we spent playing City of Winter. With lightweight rules but a heavy emphasis on the kind of ephemera usually
reserved for much crunchier games, creating space for this game is almost as important as who you chose to play it with. Characters travel along a cloth map
that illustrates a landscape with only barest indications of topography; the focus instead relies on themes of assimilation, diaspora, and — most importantly to
the narrative — baggage. As a family of immigrants (bound by blood or by choice), players describe their home, their traditions, and then, as a darkness comes
from the west side of the map, they leave it all behind, choosing what traditions and rituals they pick up from the places they visit and what they hold on to
from their homeland.
</item 90>

<item 91>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
While the playing habits of a Twitch streamer don’t represent the average player by any means, content creators certainly shape public opinion about games.

“Negative sentiment is very profitable,” says the AAA game developer, who noted that social media platforms are built to highlight popular — and sometimes
caustic — content. This dynamic sets up an inherent tension between game developers and hardcore players like Twitch streamers, who are not the main audience of
most games but are often some of the most visible players.

“I don’t at all think that streamers are embodying some typical player behaviors, like ‘Brandon,’” the developer continues, posing a name for an imaginary
everyday consumer. “There’s definitely a model of consumers nowadays who probably parrot those sentiments. They did not arrive at those conclusions themselves.”

“There is inherently [a] vicious and negative cycle for everyone involved, for how live service interacts with Twitch as a medium”

The misinformation will vary, but Twitch both shapes that discourse while also becoming the very metric that people use to make sense of a game’s situation.
</item 91>

<item 92>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The game is utterly devoted to giving the players the feeling of playing a hero in the Marvel universe with tons of cool powers and power categories based on
key Marvel heroes like Spider-Man and the Hulk. The system is easy, streamlined, and really carries the four-color comic book flavor. Probably one of the
coolest aspects of the game is that while there is an exhaustive list of Marvel heroes and villains at the back of the book, you can create your own hero with
their own power set and use that “bestiary” of heroic and villainous figures as allies and enemies in your own heroic adventures. For those of us who’ve always
wanted to make their own X-Gene Mutant, Spirit of Vengeance, or Spidersona, this is a must-have book that will bring hours of enjoyment.

Masks of the Masks

Superhero comics, for all their many flaws, remain my life’s great passion.
</item 92>

<item 93>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
“There [are] these enormous benefits for some people, but there’s also just, like, a lot of it just really grinds you down on both sides of the table,” he said,
referring to game developers and Twitch streamers.

It was inevitable for things like sleep streams, uncapped subathons, and randomly generated AI shows to materialize on Twitch; content creators feel like they
need to keep going, always

The burnout phenomenon has been at the forefront of Twitch’s public image, in large part because it doesn’t spare even the most successful creators on the
platform. Twitch’s most famous woman, Imane “Pokimane” Anys, started streaming while she was still a teen. Then, it was just a hobby. Nearly a decade later,
Anys grew to have one of the most impressive stints on the platform. Late in 2022, however, came a big change. She informed fans she would no longer stream all
day, nor would games be her primary focus anymore.
</item 93>

<item 94>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
That’s why, of all the games I’ve played this year, the one I want to shout about from the rooftops is Masks of the Masks by Hazel Amber Goswick. The game is an
incredible piece of art: a tabletop RPG presented as a fully illustrated comic book, an homage to Bronze Age superhero stories just as Watchmen did, that
critically explores and examines the violence inherent in its medium. It’s an absolute joy to behold as you flip through it; you can feel the love,
consideration, and attention to detail dripping off of every move and every panel. Then, in play, it’s a revelation; a brilliant blend of Belonging Outside
Belonging and Powered by the Apocalypse that builds on some of the luminaries of both systems to create something that just sings. Plus, it has some of the most
wicked advancement mechanics I’ve run into in years, that both add to the play experience while heightening the ever-looming sense of dread and tragedy that’s
so essential to comics of that time period. Cannot recommend enough.
</item 94>

<item 95>
Cell phone shocker as 97% of kids use their device during school hours and beyond, says study (FOX News - Health, 2023-10-01T09:05:26+00:00)
Of the 85 participants under age 13, 68% used social media and had at least one app rated "teen" or higher – allowing access to "age-inappropriate experiences,"
the study suggested.

TikTok was the most popular app used by kids aged 11 and 12, while nearly half (45%) of participants used apps with mature or adult-only ratings, such as porn
sites, fantasy sports and betting apps, according to the study findings.

UNDERAGE DRINKING DANGERS: THESE ARE THE STATES WITH THE HIGHEST RATES OF TEEN ALCOHOL USE, STUDY FINDS

More than two-thirds of all participants admitted they sometimes or often find managing their use of technology difficult — or that they use it to escape
"sorrow" or "negative feelings."

They also said they've missed sleep due to being on their phones at night.

"People feel good and are enhanced by interactions with others … Those feelings cannot be replicated by connecting through a phone."
</item 95>

<item 96>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
In Alan Wake 2, everyone’s story is happening at once, branching out in endless directions, and we might not know the genre until it’s too late. It’s a
hypertext mystery, an ergodic game that molds itself to an audience raised online, though few computers are in sight. It’s a game for a world suspended in an
endless second act — forever wars, live services, franchises, infinite scrolls — endlessly searching for a conclusion. —JR

2. Baldur’s Gate 3

Developer: Larian Studios

Where to play: PlayStation 5 and Windows PC

Baldur’s Gate 3 hit at the perfect moment. Despite the handful of truly great computer role-playing games that have been released since the early 2000s (Larian
Studios itself made waves with its Divinity: Original Sin series in 2014 and 2017), the genre has largely remained a modern niche. But times have changed:
Dungeons & Dragons has undergone something of a renaissance, thanks to actual-play series like Critical Role and Dimension 20.
</item 96>

<item 97>
Fortnite is the latest game to entice players with a portal to the past (TechCrunch, 2023-10-30T19:18:39+00:00)
In its heyday, Fortnite was about as ubiquitous and popular as a game can be. Streaming gameplay routinely drew hundreds of thousands of viewers on Twitch,
where a cottage industry of pro Fortnite players emerged, all laser-focused on Epic’s polished battle royale. Epic keeps its metrics under wraps, but back in
2020 the game had more registered players than the population of the United States.

Epic’s hit game has both imported and exported popular culture since its launch in 2017. Like other live service online games, Fortnite releases new content
every three-ish months, refreshing the destinations scattered across its cartoony island locale, changing the rules of physics and generally adding additional
mayhem, often in the form of weapons.

Those seasons are punctuated by flashy concerts and pop culture tie-ins, like a psychedelic Ariana Grande show, an in-game lightsaber fight with Darth Vader or
the appearance of Dragon Ball Z’s Goku as an in-game skin.
</item 97>

<item 98>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
Suddenly, “live service” was everywhere — and games like Overwatch, which couldn’t achieve a similar pace, were posed as disappointments by content creators.

Does an average player truly need their favorite game to give them something new, week in and week out? Developers say getting half of their players to actually
finish games is unusual and worth celebrating, because the reality is that most people will never finish a game they start. Twenty-two percent of purchased
games on Steam are unplayed outright. A number of things affect these percentages, like being a part of a sale or Game Pass, but the statistics still paint a
picture. A reported 25% of players between the ages of 18 and 34 enjoy games between three and six hours a week, 28% play two hours or less, and 19% play
between 13 to 24 hours or more, according to Statista. As they get older, fans apparently spend less time in games. In the most engaged age group, half of all
players only log a couple of hours or so per week in their favorite games.
</item 98>

<item 99>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The second, just launched, is the wild Season of Discovery, which remixes and restructures the original WoW experience — interpolating staggered level caps,
shuffling class roles — in ways that might just change MMO design forever. WoW Classic is quietly, and paradoxically, where Blizzard is doing its most
forward-thinking work right now. —Oli Welsh

39. Fire Emblem Engage

Developer: Intelligent Systems

Where to play: Nintendo Switch

Fire Emblem Engage was designed for a very specific kind of sicko: one not particularly interested in the origin stories of a horde of teenagers, or the
politics of a bourgeoise academy, or what kind of tea a teacher prefers, but instead one obsessed with the endless minutiae of combat stats, weapon loadouts,
and team composition. I know this because I am one such sicko.

If you’ve read any of my reviews or essays on Polygon, then you know I prefer strategy games that can get out of their own way.
</item 99>

<item 100>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
A serious Twitch streamer who goes live daily or logs a full workday during their schedule will typically play much, much more than the people who are watching
them. Twitch streamers can’t stop; a mere two-day pause might mean losing thousands of paying subscribers. There’s a big incentive to keep up with live-service
updates as well. There’s some FOMO involved; while certain events with special prizes happen yearly, other big occasions, like season finales, only happen once.
But the numbers don’t lie: Your viewership will definitely be impacted by new patches that float the game in question to Twitch’s front page.

“Multiplayer games will start decreasing in viewership after the first week as the launch hype dies down and then it’s a question of how good the game is
perceived to be that will determine its long term fate,” says StreamElements’ public relations director Chase. “With those in for the long haul, viewership
tends to plateau and then spike when new content and patches are released or if it is supported by competitive events.”
</item 100>
=== END ITEM id=8b76f6051dc8ce5d ===
