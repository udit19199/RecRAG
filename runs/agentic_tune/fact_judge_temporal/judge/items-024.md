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

=== ITEM id=597f7f82bad15d2f ===
Gold sentences and facts:
Sentence 0: "We all found ourselves at home looking at the people we worked with, the people we went out for drinks with and the people in our extended family on a computer screen," he told Fox News Digital.
  Fact 0.0: The speaker told Fox News Digital that everyone found themselves at home looking at the people they worked with on a computer screen.
  Fact 0.1: The speaker told Fox News Digital that everyone looked at the people they went out for drinks with on a computer screen.
  Fact 0.2: The speaker told Fox News Digital that everyone looked at the people in their extended family on a computer screen.
Sentence 1: We used to play in person, but since the pandemic I run the game virtually online from my house.
  Fact 1.0: The speaker used to play the game in person.
  Fact 1.1: Since the pandemic, the speaker runs the game virtually online.
  Fact 1.2: Since the pandemic, the speaker runs the game from the speaker's house.

Retrieved items:
<item 1>
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

Graph facts:
- Truth Behind Jake Paul Dating Dillon Danis’ Ex-girlfriend: Did MMA Star Break Up With Savannah Montano? -MENTIONS-> FaceTime
- Beeper reversed engineered iMessage to bring blue bubble texts to Android users -MENTIONS-> FaceTime
- The Ray-Ban Meta smart glasses actually make the future look cool -MENTIONS-> FaceTime
- After losing her leg to the flu, Virginia woman urges people to get vaccinated: ‘Don’t waste time’ -PUBLISHED_BY-> FOX News - Health
- Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study -PUBLISHED_BY-> FOX News - Health

Entities: FaceTime, FOX News - Health, Zoom, ‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts, Elliott, Zoom
fatigue, video platforms, Corcoran, social media
</item 1>

<item 2>
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

Graph facts:
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Blizzard
- Diablo 4 season 2’s biggest draw is its quality-of-life updates -MENTIONS-> Blizzard
- Fortnite is the latest game to entice players with a portal to the past -MENTIONS-> Blizzard
- Blizzard -SAID-> experience gains
- Elizabeth Harper -SAID-> Blizzard

Entities: Blizzard, Oli Welsh, Season of Discovery, WoW, MMO design, Fire Emblem Engage, Intelligent Systems, strategy games, The best video games of the year
so far, Polygon, Nintendo Switch, WoW Classic
</item 2>

<item 3>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
It is not as much of a heroic power fantasy as D&D 5e or Pathfinder, but neither is it as minimalist as many old-school games. With fickle gods, mutating
magics, lucky scoundrels, and ferocious fighters, DCC feels like playing through a Bal-Sagoth or Eternal Champion song. It’s also easy to pick up if your group
has any experience with D&D’s third edition or Pathfinder. DCC rocks! Its focus on simple, impactful rules was a core design inspiration for the (almost
system-neutral) Survivalist’s Guide to Spelunking for 5e, a book I wrote with Douglas Niles and Stephen Yeardley. It warms my heart to know that DCC authors,
judges, and players use the book as well!

Dungeons & Dragons

Mike Haracz, corporate R&D chef and host of Heroes’ Feast

I’m lucky enough to be in the seventh year of a homebrew Dungeons & Dragons campaign, one I play with friends locally in the Chicagoland area.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- 9 great movies to watch on Tubi for free right now -PUBLISHED_BY-> Polygon
- The future of DC movies in 2022 and beyond -PUBLISHED_BY-> Polygon
- The best movies of 2023 so far -PUBLISHED_BY-> Polygon
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon

Entities: Polygon, Dungeons & Dragons, tabletop RPGs, D&D 5e, Pathfinder, DCC, Bal-Sagoth, Eternal Champion, Survivalist’s Guide to Spelunking for 5e, Douglas
Niles, Stephen Yeardley, Mike Haracz, Heroes’ Feast, homebrew Dungeons & Dragons campaign, Chicagoland area
</item 3>

<item 4>
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

Graph facts:
- There’s something going on with AI startups in France -MENTIONS-> mental health
- The obsessive tormenter who made professors’ lives miserable -MENTIONS-> mental health
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -MENTIONS-> mental health
- Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes -MENTIONS-> mental health
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> mental health

Entities: mental health, anxiety, the pandemic, FOX News - Health, Fox News Digital, ‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle
it, according to experts, Nassar, Elliott, Zoom fatigue, remote workers, video-ing of each other, video calls, stress, burnout, in-person interactions
</item 4>

<item 5>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Our campaign is called The Heartsguard Saga, and it’s based off Deven Rue’s map of a place called Euphoros. My players are at around level 15 and I frequently
give them a variety of amazing and unique gear that I’ve dreamed up all on my own... which they go on to forget they have, and never use. We used to play in
person, but since the pandemic I run the game virtually online from my house. I put a camera hovering over my gaming mat, and make a labeled grid for the
players so they call out, “Move me from A5 to B10.” During important rolls, I even make my players face their camera toward their dice for added suspense!

Petrana Radulovic, entertainment reporter

I’d wanted to play Dungeons & Dragons since I was in high school, but I never knew how to get into it.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- 9 great movies to watch on Tubi for free right now -PUBLISHED_BY-> Polygon
- The future of DC movies in 2022 and beyond -PUBLISHED_BY-> Polygon
- The best movies of 2023 so far -PUBLISHED_BY-> Polygon
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon

Entities: Polygon, pandemic, Dungeons & Dragons, tabletop RPGs, The Heartsguard Saga, Deven Rue, Euphoros, The best tabletop RPGs we played in 2023, Petrana
Radulovic
</item 5>

<item 6>
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

Graph facts:
- After losing her leg to the flu, Virginia woman urges people to get vaccinated: ‘Don’t waste time’ -PUBLISHED_BY-> Fox News
- Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study -PUBLISHED_BY-> Fox News
- Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' -PUBLISHED_BY-> Fox News
- Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' -PUBLISHED_BY-> Fox News
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -PUBLISHED_BY-> Fox News

Entities: Fox News, FOX News app, Zoom, ‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts, Zoom fatigue,
burnout, Corcoran, mental health outcomes, chronic stress
</item 6>

<item 7>
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

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> San Francisco
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> San Francisco
- Here’s how Rainforest, a budding Stripe rival, aims to win over software companies -MENTIONS-> San Francisco
- 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit -MENTIONS-> San Francisco
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> San Francisco

Entities: San Francisco, FOX News - Health, Fox News Digital, COVID-19 pandemic, Julia Corcoran, Modern Health, Carl Nassar, Denver, Colorado, Zoom,
videoconferencing platforms, ‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts, Zoom fatigue
</item 7>

<item 8>
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

Graph facts:
- Cranberries: A Thanksgiving Staple With a Rich History and Unique Biology -MENTIONS-> animals
- Dangerous 'superbugs' are a growing threat, and antibiotics can't stop their rise. What can? -MENTIONS-> animals
- 'They seemed primed to take over': How the Great Dying doomed the 'beast tooth' and set the stage for the dawn of the dinosaurs -MENTIONS-> animals
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon

Entities: animals, Polygon, David Attenborough, The best tabletop RPGs we played in 2023, Dixon, Exquisite Biome, Everest Pipkin, Caro Asercion, Si Sweetman,
Spore, niche online community, group chat, centralized internet, world-building TTRPG, ecological biomes
</item 8>

<item 9>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Thankfully, in the doldrums of 2020, someone I knew knew someone who knew someone (that’s literally what the connection was) who was interested in running a
game. That kicked off my D&D career, but was this year that everything finally clicked in my brain and it went from a hobby I enjoyed to one that I spend every
free minute of my spare time thinking about.

Part of it is finally having reliable groups. After some initial shuffling of members, in 2023 I’ve been blessed to have not one but two separate D&D groups,
both with enthusiastic members who are committed to at least trying to meet regularly. And yes, part of it is Baldur’s Gate 3, which helped me to understand
some of the more complicated D&D mechanics and just generally made me very excited about combat in a way I never was before.

I’ve still yet to actually have a game in person! My groups are spread out across North America, but I’m hoping to manifest an in-person session soon.

Graph facts:
- How to choose the best class for you in Baldur’s Gate 3 -MENTIONS-> D&D
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- 9 great movies to watch on Tubi for free right now -PUBLISHED_BY-> Polygon
- The future of DC movies in 2022 and beyond -PUBLISHED_BY-> Polygon

Entities: D&D, Polygon, Baldur’s Gate 3, North America, tabletop RPGs, The best tabletop RPGs we played in 2023
</item 9>

<item 10>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Players take on the role of soldiers and other hangers-on just trying to make it out alive. The heady mix of dice-driven gunplay and a playing-card-style
encounter creation makes it easy to run. But the fun begins even before that with character generation, which uses old school randomized tables to simulate an
entire military career. If you’re looking to get your video gaming buddies around the table for something that doesn’t require a headset or shouting down
pre-teens in a glitchy public lobby, the all-inclusive starter set comes highly recommended.

Project ECCO

It’s a near-impossible task to get a consistent group together, so solo TTRPGs have become my primary method of gaming. However, the barrier for me is finding a
game with a balance of narrative and mechanics that isn’t overwhelmingly crunchy, but also isn’t open-ended to the point of choice paralysis. For me, Project
ECCO found that balance.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon
- 9 great movies to watch on Tubi for free right now -PUBLISHED_BY-> Polygon
- The future of DC movies in 2022 and beyond -PUBLISHED_BY-> Polygon
- The best movies of 2023 so far -PUBLISHED_BY-> Polygon
- How to choose the best class for you in Baldur’s Gate 3 -PUBLISHED_BY-> Polygon

Entities: Polygon, The best tabletop RPGs we played in 2023, Project ECCO
</item 10>
=== END ITEM id=597f7f82bad15d2f ===

=== ITEM id=286781115cbcb9ee ===
Gold sentences and facts:
Sentence 0: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 0.0: Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 0.1: The matter referred to as 'this' will probably come up later in the trial.
Sentence 1: But it didn’t release the full model, Gemini Ultra — only a “lite” version called Gemini Pro.
  Fact 1.0: Google did not release the full model, Gemini Ultra.
  Fact 1.1: Google released only a 'lite' version called Gemini Pro.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content, their readers and ad revenue through anticompetitive means.

Retrieved items:
<item 1>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Sundar Pichai
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Sundar Pichai
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Sundar Pichai
- 5 things we learned so far about the Google antitrust case -MENTIONS-> Sundar Pichai
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Sundar Pichai

Entities: Sundar Pichai, Android, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic Games, Wilson White, Match,
U.S. attorneys general, Epic's antitrust lawsuit, U.S. District Court in California’s Northern District, Department of Justice, Google's antitrust lawsuit with
the Department of Justice, search monopoly, antitrust, Netflix, Apple
</item 1>

<item 2>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Both Epic and Google spent a long, long time on subtle lines of questioning. You really had to read between the lines to see that Epic was trying to make a
point about how Google’s 70/30 revenue split is probably based on an arbitrary decision Valve made two decades ago with Steam or how Google was trying to make a
point that Epic, too, likely believed that an app store provides more value than just payment processing and maybe deserves more money.

Graph facts:
- The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more -MENTIONS-> App Store
- Sarah Michelle Gellar hits out at Roblox after her son’s account gets hacked -MENTIONS-> App Store
- Best Golf Betting Sites and Apps - Top Sportsbooks for Golf 2023 -MENTIONS-> App Store
- The best video games of the year so far -MENTIONS-> App Store
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> App Store

Entities: App Store, Epic, Steam, Google, The Epic v. Google trial may come down to simple v. complicated, Epic v. Google trial, 70/30 revenue split, payment
processing, Valve, The Verge
</item 2>

<item 3>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- 5 things we learned so far about the Google antitrust case -MENTIONS-> FTC-Facebook case
- Amit Mehta -PARTICIPATED_IN-> US v Google trial

Entities: Amazon antitrust lawsuit, FTC-Facebook case, Google antitrust lawsuit with Match Group, Google trial with Epic Games, Anticompetitive and unfair
strategies to maintain monopoly power, Vertical search, Pre-installs, Third-party apps, Commission structure, More from the US v Google trial: vertical search,
pre-installs and the case of Firefox/Yahoo, US v Google trial, Yahoo, Firefox, anticompetitive behavior, Google Play
</item 3>

<item 4>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Or at least it was worse, before they were destroyed.”

The best Google could do in response was to plant its own feeble seed with the jury, too: “Is Epic using the chats to distract me from all the evidence I do
see?”

“It’s true that Google could have automatically saved all chats for all relevant employees, but just because Google didn’t save some chats didn’t mean it
violated antitrust laws,” Pomerantz argued.

Epic’s opening statements seemed to paint a clearer picture for the jury than those from Google. But things got complicated for both parties when the first two
witnesses — Epic Games Store head Steve Allison and Yoga Buddhi CEO Benjamin Simon, who also appeared in the earlier Epic v. Apple trial — took the stand.

Both Epic and Google spent a long, long time on subtle lines of questioning.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Epic
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Epic
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Epic
- Epic v. Google, explained -MENTIONS-> Epic
- Can anyone survive Fortnite as a job? -MENTIONS-> Epic

Entities: Epic, Epic Games Store, Google, The Epic v. Google trial may come down to simple v. complicated, Pomerantz, Epic v. Google trial, Steve Allison,
Benjamin Simon, Yoga Buddhi, antitrust laws, Epic v. Apple trial, The Verge
</item 4>

<item 5>
Graph query row:
article: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line
publisher: TechCrunch
published_on: 2023-12-15T17:56:02+00:00
Google: Google
article_connection: MENTIONS
Google_claim_relationship: ACCUSED_OF
claim: US v Google trial
connections: [{'entity': 'Alphabet', 'relationship': 'MENTIONS'}, {'entity': 'Helena World Chronicle', 'relationship': 'MENTIONS'}, {'entity': 'U.S. District
Court in D.C.', 'relationship': 'MENTIONS'}, {'entity': 'Search Generative Experience', 'relationship': 'MENTIONS'}, {'entity': 'Bard AI chatbot',
'relationship': 'MENTIONS'}, {'entity': 'Knowledge Graph', 'relationship': 'MENTIONS'}, {'entity': 'anticompetitive behavior', 'relationship': 'MENTIONS'},
{'entity': 'U.S. antitrust law', 'relationship': 'MENTIONS'}, {'entity': 'Sherman Act', 'relationship': 'MENTIONS'}, {'entity': 'Arkansas', 'relationship':
'MENTIONS'}, {'entity': 'question-and-answer technologies', 'relationship': 'MENTIONS'}, {'entity': 'Publishers', 'relationship': 'MENTIONS'}, {'entity':
'Knowledge Panel', 'relationship': 'MENTIONS'}, {'entity': 'Featured Snippets', 'relationship': 'MENTIONS'}, {'entity': 'class action antitrust suit',
'relationship': 'MENTIONS'}, {'entity': 'misappropriation of Publishers’ content', 'relationship': 'MENTIONS'}, {'entity': 'The Wall Street Journal',
'relationship': 'MENTIONS'}, {'entity': 'report on Thursday', 'relationship': 'MENTIONS'}, {'entity': 'deal to license news for AI model training',
'relationship': 'MENTIONS'}, {'entity': 'SGE', 'relationship': 'MENTIONS'}, {'entity': 'Google Bard AI', 'relationship': 'MENTIONS'}, {'entity': 'GoogleBot',
'relationship': 'MENTIONS'}, {'entity': 'Google general search service', 'relationship': 'MENTIONS'}, {'entity': 'AdSense', 'relationship': 'MENTIONS'},
{'entity': 'News Media Alliance', 'relationship': 'MENTIONS'}, {'entity': 'Washington Post', 'relationship': 'MENTIONS'}, {'entity': 'Epic Games lawsuit against
Google over app store antitrust issues', 'relationship': 'MENTIONS'}, {'entity': 'app store antitrust issues', 'relationship': 'MENTIONS'}, {'entity': 'U.S.
Justice Department', 'relationship': 'MENTIONS'}, {'entity': 'Hausfeld', 'relationship': 'MENTIONS'}, {'entity': 'Helena World Chronicle, LLC', 'relationship':
'MENTIONS'}, {'entity': 'class action antitrust suit against Google', 'relationship': 'MENTIONS'}, {'entity': 'U.S. Justice Department lawsuit against Google',
'relationship': 'MENTIONS'}, {'entity': '2020 Justice Department civil antitrust suit over search and search advertising', 'relationship': 'MENTIONS'},
{'entity': 'digital ad technologies', 'relationship': 'MENTIONS'}, {'entity': 'Clayton Act', 'relationship': 'MENTIONS'}, {'entity': 'search and search
advertising', 'relationship': 'MENTIONS'}, {'entity': 'digital news and reference publishing', 'relationship': 'MENTIONS'}, {'entity': 'class-wide monetary and
injunctive relief', 'relationship': 'MENTIONS'}, {'entity': 'free marketplace of ideas', 'relationship': 'MENTIONS'}, {'entity': 'free press', 'relationship':
'MENTIONS'}, {'entity': 'Epic Games', 'relationship': 'MENTIONS'}, {'entity': 'antitrust', 'relationship': 'MENTIONS'}, {'entity': 'The Atlantic',
'relationship': 'MENTIONS'}, {'entity': 'Axel Springer', 'relationship': 'MENTIONS'}, {'entity': 'Allen Institute for AI', 'relationship': 'MENTIONS'},
{'entity': 'class action lawsuit', 'relationship': 'MENTIONS'}, {'entity': 'consumers', 'relationship': 'MENTIONS'}, {'entity': 'AI', 'relationship':
'MENTIONS'}, {'entity': 'AI technologies', 'relationship': 'MENTIONS'}, {'entity': 'Similarweb', 'relationship': 'MENTIONS'}, {'entity': 'Google',
'relationship': 'MENTIONS'}, {'entity': 'artificial intelligence', 'relationship': 'MENTIONS'}, {'entity': 'D.C.', 'relationship': 'MENTIONS'}, {'entity':
'OpenAI', 'relationship': 'MENTIONS'}, {'entity': 'labor', 'relationship': 'MENTIONS'}]
passages: ['News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)\nA
new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in violation
of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle, argues
that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI technologies
like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.\n\nIn the complaint, Helena World Chronicle, which owns and
publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing publishers’ content on Google, losing them “billions of
dollars.”\n\nIn addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May
2012, as part of the problem.', "News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch,
2023-12-15T17:56:02+00:00)\n“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel
contains a summary of content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information
from Publishers’ websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.\n\nBy 2020, the
Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was content
“misappropriated from Publishers,” the complaint alleges.\n\nOther Google technologies, like “Featured Snippets” where Google algorithmically extracts answers
from webpages, were also cited as shifting traffic away from publishers’ websites.\n\nMore importantly, perhaps, is the suit’s tackling of how AI will impact
publishers’ businesses. The problem was recently detailed in a report on Thursday by The Wall Street Journal, which led with a shocking statistic.", 'News
publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)\nWhen online magazine
The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s query without requiring
a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today drives nearly 40% of
their traffic, according to data from Similarweb.\n\nSome publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked
a deal with OpenAI to license its news for AI model training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic
when Google’s AI products fully roll out, The WSJ’s report noted.\n\nThe lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based
search were implemented with “the goal of discouraging end-users from visiting the websites of Class members who are part of the digital news and publishing
line of commerce.”', 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch,
2023-12-15T17:56:02+00:00)\nSGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s
“walled garden” as it “plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service,
GoogleBot.\n\nPlus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from
the News Media Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for
AI, had found that News and Media sites were the third largest category of AI training data.)\n\nThe case points to other concerns, too, like changing AdSense
rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat messages — an issue raised in the recent Epic Games lawsuit
against Google over app store antitrust issues, which Epic won.', 'News publisher files class action antitrust suit against Google, citing AI’s harms to their
bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)\nThe case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for
monopolizing digital ad technologies, and references the 2020 Justice Department’s civil antitrust suit over search and search advertising (which are different
markets from digital ad technologies in the more recent suit).\n\n“The anticompetitive effects of Google’s scheme cause profound harm to competition, to
consumers, to labor, and to a democratic free press,” reads an announcement posted to the website of the law firm handling the case, Hausfeld.\n\n“Plaintiff
Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure competition for
digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,” it states.\n\nA
Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.']
</item 5>

<item 6>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android

Entities: Android, Apple, Epic, Google, The Epic v. Google trial may come down to simple v. complicated, Pomerantz, Project Liberty, Android sideloading flow,
app store fees, Google policy, app stores, Epic v. Google trial, Fortnite, The Verge
</item 6>

<item 7>
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

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Project Hug
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Project Hug
- Tim Sweeney -SAID-> Project Hug
- Project Hug -PART_OF-> Google Games Velocity Program
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Epic

Entities: Project Hug, Epic, Google, Epic v. Google, explained, Sweeney, state attorneys general, Project Agave, Project Banyan, game developers, phone makers,
Google's app store, settlement, anticompetitive practices, paying game developers and phone makers not to abandon Google's app store, deleting potentially
incriminating messages
</item 7>

<item 8>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> User Choice Billing
- The fate of US venture capital in China teeters on uncertainty -MENTIONS-> Tencent
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Tencent
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Tencent
- European neobroker Scalable Capital raises $65M on a flat $1.4B valuation -MENTIONS-> Tencent

Entities: User Choice Billing, Tencent, Epic, China, Google, Epic v. Google, explained, Epic v. Google lawsuit, Judge Donato, Samsung Galaxy phone, The Verge
</item 8>

<item 9>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Google Play
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Google Play
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Google Play
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Google Play
- Beeper Mini is back in operation after Apple’s attempt to shut it down -MENTIONS-> Google Play

Entities: Google Play, Activision, Epic, Google, The Epic v. Google trial may come down to simple v. complicated, app store and payment processing fees, Epic v.
Google trial, competition, The Verge
</item 9>

<item 10>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Epic Games
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Epic Games
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Epic Games
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Epic Games
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Epic Games

Entities: Epic Games, Sean, Apple, Google, Epic v. Google trial, Epic v. Apple trial, #FreeFortnite, Llamacorn, antitrust claims, illegal monopoly, Epic v.
Google, explained, Fortnite, The Verge
</item 10>
=== END ITEM id=286781115cbcb9ee ===

=== ITEM id=3ef7c675fe7e9387 ===
Gold sentences and facts:
Sentence 0: In the long term, scientists also point to a number of recent developments which have raised hopes that it may be possible to tackle dementia more directly and effectively.
  Fact 0.0: Scientists also point to a number of recent developments in the long term.
  Fact 0.1: The recent developments have raised hopes that it may be possible to tackle dementia more directly and effectively.
Sentence 1: Health coach Susan Saunders found herself “parenting” her mother (who had dementia) while her own daughters were babies.
  Fact 1.0: Health coach Susan Saunders found herself 'parenting' her mother.
  Fact 1.1: Susan Saunders's mother had dementia.
  Fact 1.2: Susan Saunders's daughters were babies at the time.

Retrieved items:
<item 1>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“It is a challenge, and a very important one.”

Graph facts:
- Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma -MENTIONS-> Alzheimer’s
- David Thomas -SAID-> Alzheimer’s
- adacanumab -PART_OF-> Alzheimer’s
- Lecanemab -PART_OF-> Alzheimer’s
- Manchester United face Galatasaray with high hopes but bad memories -PUBLISHED_BY-> The Guardian

Entities: Alzheimer’s, The Guardian, New Alzheimer’s drugs bring hope of slowing disease for UK patients, UK
</item 1>

<item 2>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
“Anyone with older parents should be urging them out for a walk five days a week – and when they have a companion, the walk also doubles as a social occasion,”
he explained. “If they can walk for longer than 20 minutes, even better. Our study showed that the risk of depression fell in proportion to the time spent
walking.”

As well as a daily walk, Laird recommends vitamin D and B12 supplements – both of which appear to moderate the chance of depression in older people. “Depression
matters,” he added. “Not just because it reduces quality of life, but because in older people there seems to be a link between depression and dementia which
we’re still unpacking.”

Health coach Susan Saunders found herself “parenting” her mother (who had dementia) while her own daughters were babies. “I’d advise anyone with a parent in the
early stages of dementia to investigate all the available research trials. Start by joining Dementia Research, a nationwide register that matches volunteers to
research programmes.”

Graph facts:
- ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir -PUBLISHED_BY-> The Guardian
- West Indies v England: first T20 international – live -PUBLISHED_BY-> The Guardian
- West Indies v England: fifth and deciding men’s T20 cricket international – live -PUBLISHED_BY-> The Guardian
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -PUBLISHED_BY-> The Guardian
- Manchester United face Galatasaray with high hopes but bad memories -PUBLISHED_BY-> The Guardian

Entities: The Guardian, dementia, depression, walking, Laird, Susan Saunders, vitamin D supplements, B12 supplements, research trials, Dementia Research,
research programmes, Walks, tech and protein: how to parent your own parents
</item 2>

<item 3>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
In the long term, scientists also point to a number of recent developments which have raised hopes that it may be possible to tackle dementia more directly and
effectively. One major challenge they have faced is the problem of getting drugs to pass through the blood-brain barrier, which controls the movement of ions
and molecules from our bodies to our brains. That makes it difficult to get drugs into the central nervous system to tackle problems such as amyloid plaques.

“However, researchers are developing active transport methods for getting drugs across the blood-brain barrier,” said Mummery. “For example, we are now
exploring ways of helping drugs get into the brain much more efficiently, and that could have a major impact.”

Such developments will take years to realise, scientists caution, and a great deal needs to be done in the short term to deal with dementia.

“It is clear we have taken a step in the right direction but there is much more to do before we can have successfully dealt with these processes that are
happening to our brains,” said Thomas.

Graph facts:
- Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma -MENTIONS-> Alzheimer’s
- David Thomas -SAID-> Alzheimer’s
- adacanumab -PART_OF-> Alzheimer’s
- Lecanemab -PART_OF-> Alzheimer’s
- Manchester United face Galatasaray with high hopes but bad memories -PUBLISHED_BY-> The Guardian

Entities: Alzheimer’s, The Guardian, Thomas, New Alzheimer’s drugs bring hope of slowing disease for UK patients, Mummery, dementia, blood-brain barrier, active
transport methods, central nervous system, amyloid plaques, Alzheimer’s drugs, UK
</item 3>

<item 4>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
Until now doctors have only been able to prescribe medicines that help patients manage their symptoms, so the arrival of the first drugs that treat the actual
cause of the condition has been welcomed – although experts have warned that their use should be treated with some caution.

After decades of research, these are the first drugs to improve patients’ lives directly. If nothing else, they suggest we’re on the right road to tackling
Alzheimer’s

“The new drugs slow down the development of Alzheimer’s by six months to a year and are useful only for those in the early stages of the condition, so they are
certainly not miracle medicines,” said David Thomas, head of policy at Alzheimer’s Research UK.

“However, after decades of research, they are the first to improve patients’ lives directly, and that is a justifiable cause for excitement. If nothing else,
they suggest we are probably on the right road to tackling Alzheimer’s.”

This point was backed by neurologist Cath Mummery, of the Dementia Research Centre at University College London.

Graph facts:
- Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma -MENTIONS-> Alzheimer’s
- Manchester United face Galatasaray with high hopes but bad memories -PUBLISHED_BY-> The Guardian
- adacanumab -PART_OF-> Alzheimer’s
- Lecanemab -PART_OF-> Alzheimer’s
- ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir -PUBLISHED_BY-> The Guardian

Entities: Alzheimer’s, The Guardian, New Alzheimer’s drugs bring hope of slowing disease for UK patients, David Thomas, Alzheimer’s Research UK, Cath Mummery,
Dementia Research Centre, University College London
</item 4>

<item 5>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
People in Britain could benefit from a key medical breakthrough next year. They may be given access to the first drugs ever developed to slow the impact of
Alzheimer’s disease.

The first of these medicines – lecanemab – was recently approved in the US and Japan, where treatments using it have already been launched. A second drug,
donanemab, is expected to follow soon, and next year it is anticipated that the UK medical authorities will consider both of them for approval in Britain.

The prospect has raised hopes that, after years of effort, scientists may be closing in on ways to directly tackle the UK’s dementia crisis. About a million
people are living with the condition in this country, and this is expected to rise to about 1.7 million by 2040 – with potentially grim consequences. Last year
dementia took the lives of 66,000 people in England and Wales, and it is now the leading cause of death in Britain, with Alzheimer’s accounting for two-thirds
of cases.

Graph facts:
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> Japan
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Japan
- Discogs’ vibrant vinyl community is shattering -MENTIONS-> Japan
- MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency -MENTIONS-> Japan
- At 37, Alexandra found herself broke, heartbroken and living with her parents -MENTIONS-> Japan

Entities: Japan, US, Alzheimer’s disease, The Guardian, New Alzheimer’s drugs bring hope of slowing disease for UK patients, UK’s dementia crisis, England and
Wales, UK medical authorities, Lecanemab, donanemab, dementia, Britain, UK
</item 5>

<item 6>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
It took 10 years for Mike Mackenzie-Smith to get a diagnosis of young-onset Alzheimer’s.

Another grim illustration of the problem facing patients is provided by Graeme Armstrong. His wife, Trina, began having trouble recognising faces and reading
telephone numbers in 2006 and was told, after a CT scan three years later, that she had probably had a stroke – although her symptoms did not suggest such a
diagnosis. It took another three years before she was diagnosed with posterior cortical atrophy, a rare variant of Alzheimer’s disease that affects the way the
brain interprets information from the eyes.

“Had we had an accurate diagnosis four years earlier, Trina could have been placed on the correct medication, which may have been more effective and helped her
day to day,” Armstrong said.

One solution sought by doctors involves the creation of blood tests which could pinpoint the disease quickly and effectively. “These are under research, but we
are several years away from having them in widespread use,” said Thomas.

Graph facts:
- Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ -MENTIONS-> Alzheimer’s disease
- Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma -MENTIONS-> Alzheimer’s disease
- Valter Longo -SAID-> Alzheimer’s disease
- Antoun -SAID-> Alzheimer’s disease
- Alzheimer’s Association -SAID-> Alzheimer’s disease

Entities: Alzheimer’s disease, The Guardian, Thomas, New Alzheimer’s drugs bring hope of slowing disease for UK patients, Mike Mackenzie-Smith, Graeme
Armstrong, Trina, young-onset Alzheimer’s, posterior cortical atrophy, stroke, CT scan, blood tests, medication
</item 6>

<item 7>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“Meanwhile, we have got to get the NHS to a state where it is much more focused and better organised in getting better and earlier dementia diagnoses. That will
be crucial in our battle against the condition.”

Mummery agreed: “By the time you have dementia, you will have had Alzheimer’s disease developing in your brain for at least 20 years. Dementia is the latest
stage; we need to pick up signals much earlier. So we need to be thinking about how to diagnose people at the very earliest stages of disease, when they may
only have very, very subtle symptoms.

We are now exploring ways of helping drugs get into the brain much more efficiently, and that could have a major impact Cath Mummery

“Our current service isn’t good for doing that, and we need to develop brain health clinics where we can pinpoint the disease much earlier, then help to build
resilience against dementia in a patient when it is only in a very early stage.”
</item 7>

<item 8>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
Both drugs are to be considered for approval in the UK next year. The Medicines & Healthcare products Regulatory Agency (MHRA) will first decide if they are
safe and effective, then the National Institute for Health and Care Excellence (Nice) will rule on whether they offer value for money.

Both medicines are expensive – lecanemab costs about $25,000 (£19,700) a year – and are given by regular intravenous infusions. “That is a challenge from a
health-service point of view, because you need to find the space and time to put someone in an infusion suite to treat them,” added Thomas.

However, the main problem facing doctors is the difficulty involved in pinpointing dementia not just in its early stages but even in its later phases. Most
cases are first presented to GPs, who will then refer patients to memory clinics for a dementia test. However there are long waiting times – up to two years on
average – for appointments at these centres.
</item 8>

<item 9>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“It has been a very long, hard road, but finally we have something positive to look at. That is very welcome.”

Alzheimer’s disease is triggered by the buildup of a protein called amyloid in the brain, although symptoms may not appear for decades after this accretion has
started. Scientists have tried for more than 20 years to find ways to stop amyloid from forming these plaques, in the hope this would stop the progression of
the disease.

Lecanemab, which is produced by Japanese pharmaceutical company Eisai, and donanemab, produced by Eli Lilly of the US, are the first medicines to achieve this
aim – though they only slow but do not ultimately halt the disease’s progression.

Graeme Armstrong with wife Trina, who has a rare form of Alzheimer’s.

Both drugs are to be considered for approval in the UK next year.

Graph facts:
- ASX set for bright start despite Wall Street slip; $A drops -MENTIONS-> US
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> US
- Is Google Search better than the rest? And is that fair? -MENTIONS-> US
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> US
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> US

Entities: US, amyloid, Alzheimer’s disease, The Guardian, New Alzheimer’s drugs bring hope of slowing disease for UK patients, Lecanemab, Eisai, donanemab, Eli
Lilly, consideration for approval, Graeme Armstrong, Trina, UK
</item 9>

<item 10>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
However there are long waiting times – up to two years on average – for appointments at these centres.

In addition, diagnosing Alzheimer’s and other forms of dementia is usually based on pen-and-paper tests, followed by lumbar punctures and brain scans before a
final diagnosis.

About 65% of cases are confirmed this way. The remaining third of dementia cases are never diagnosed. Yet patients can only expect to receive treatments –
including the new drugs – if their condition is identified.

Eleanor Mackenzie-Smith’s father, Mike, has young-onset Alzheimer’s. His first symptoms were spotted 17 years ago, when Eleanor was 11. “However, it took over
10 years from the start of his symptoms and four separate tests between 2009 and 2017 to get my dad’s final diagnosis, when he was 65. It was distressing not
knowing what was happening. Too many families, like mine, have had to watch as dementia takes hold of our loved ones while we are left without a diagnosis and
the support and access to treatments.”

Graph facts:
- Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma -MENTIONS-> Alzheimer’s
- David Thomas -SAID-> Alzheimer’s
- adacanumab -PART_OF-> Alzheimer’s
- Lecanemab -PART_OF-> Alzheimer’s
- Manchester United face Galatasaray with high hopes but bad memories -PUBLISHED_BY-> The Guardian

Entities: Alzheimer’s, The Guardian, New Alzheimer’s drugs bring hope of slowing disease for UK patients, Eleanor Mackenzie-Smith, new Alzheimer’s drugs,
dementia, UK, Mike
</item 10>
=== END ITEM id=3ef7c675fe7e9387 ===
