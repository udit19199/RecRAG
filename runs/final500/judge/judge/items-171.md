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

This file has 6 items.

=== ITEM id=07d3a847a618098c ===
Gold sentences and facts:
Sentence 0: It has also decided to make Beeper Mini free for the time being, given how “chaotic” things have been over the past few days.
  Fact 0.0: The party referred to as 'it' has decided to make Beeper Mini free for the time being.
  Fact 0.1: Things have been "chaotic" over the past few days.
  Fact 0.2: The decision to make Beeper Mini free was made given how "chaotic" things have been over the past few days.
Sentence 1: In addition, Beeper Mini will be free for the time being given the unstable nature of Beeper’s status and Apple’s attempts at shutting it down.
  Fact 1.0: Beeper Mini will be free for the time being.
  Fact 1.1: Beeper's status is unstable.
  Fact 1.2: Apple has attempted to shut Beeper Mini down.
  Fact 1.3: Beeper Mini is free because of the unstable nature of Beeper's status and Apple's attempts at shutting it down.
Sentence 2: Beeper Mini was made possible because the team behind the app has managed to reverse-engineer the iMessage protocol, they say.
  Fact 2.0: Beeper Mini was made possible because the team behind the app reverse-engineered the iMessage protocol.
  Fact 2.1: The team behind the app says it has managed to reverse-engineer the iMessage protocol.

Retrieved items:
<item 1>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
It’s $1.99/mo, not $2.99/mo; 12/6/23, 2:33 PM ET added “[sent it]” to clarify that the Beeper app does not look like an iPhone app.
</item 1>

<item 2>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
Full-res photos and other features like tapbacks and typing indicators also worked. When the Android phone’s battery died, however, the texts reverted to green
bubbles and did not make it to Beeper’s app — they went to Google Messages instead.

The company is also hoping to gain trust by building in public, with 50-plus projects that it’s published to GitHub with the open source code that goes into the
app. Plus, the founders themselves are known individuals with a history of building promising tech, including the Pebble smartwatch.

Founded in 2020, Beeper comes from former Y Combinator partner Eric Migicovsky and CTO Brad Murray, previously of wholesale marketplace startup Faire and
Fitbit. The co-founders met at Pebble, the smartwatch company Migicovsky founded, building out the hardware brand that was later acquired by Fitbit.

Migicovsky said he was inspired to build Beeper out of a personal need as a lifelong Android user.
</item 2>

<item 3>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
The push to bring iMessage to Android users today adds a new contender. A startup called Beeper, which had been working on a multi-platform messaging
aggregator, is now launching a new app called Beeper Mini which will allow Android users to send and receive end-to-end encrypted iMessage chats for just $1.99
per month. Beeper Mini was made possible because the team behind the app has managed to reverse-engineer the iMessage protocol, they say.

“That’s the big breakthrough,” explains Beeper co-founder and CEO Eric Migicovsky, previously the founder of smartwatch startup Pebble. “We’re not actually a
middleman anymore. The research that we’ve done is actually reverse-engineering the iMessage protocol, down to the lowest layer of the protocol. So Beeper Mini
doesn’t use a Mac server as a relay like all the other apps — they have a Mac Mini in a data center somewhere.
</item 3>

<item 4>
Graph query row:
article: Beeper Mini is back in operation after Apple’s attempt to shut it down
published_on: 2023-12-11T18:08:48+00:00
publisher: TechCrunch
product: Beeper Mini
</item 4>

<item 5>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
Initially, Beeper launched a multi-network chat app that supported around 15 different networks, including WhatsApp, Messenger, Telegram, Signal, Slack,
Instagram, LinkedIn, Discord, Google Chat, Android SMS, iMessage and others.

“Until I started Beeper, I didn’t really understand this whole iMessage thing. I didn’t really understand why people loved it so much,” Migicovsky says. “I
think one of the reasons is that it’s so deeply built into iOS, that it’s second nature for you as an iPhone user. When you want to get in touch with someone,
you open the Messages app.”

His experience with messaging had differed, though, having lived in Europe and Asia at times, where he collected friends across a wider network of applications.
He didn’t understand the draw that iMessage had in the U.S. in particular, as he was simply left out of friends’ iMessage chats for being the friend with a
green bubble.
</item 5>

<item 6>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
The company is publishing those results on its blog along with a detailed, more technical description of how Beeper Mini works.

For example, the team explains here how it needed to build a new service, called Beeper Push Notification service (BPNs), to make the service work:

A persistent connection to APNs is needed to be notified of new incoming messages in real-time. On an iPhone, an APNs connection is maintained by the operating
system, and connected at all times. In Beeper Mini, the connection can only be maintained when the app is running, since Android does not support APNs natively.

To work around this limitation, the team built BPNs to connect to Apple’s servers on the user’s behalf when the app isn’t running.

In tests, Beeper Mini worked as described, able to send iMessage texts from an older Android phone on Google Fi’s wireless network with its own phone number
(not associated with an Apple ID) to an iPhone 15 Pro Max with a different phone number that is associated with an Apple ID.
</item 6>

<item 7>
Beeper Mini is back in operation after Apple’s attempt to shut it down (TechCrunch, 2023-12-11T18:08:48+00:00)
Senator Elizabeth Warren (D-Mass.) shamed Apple for its intervention in a post on X, formerly Twitter, saying “Green bubble texts are less secure. So why would
Apple block a new app allowing Android users to chat with iPhone users on iMessage? Big Tech executives are protecting profits by squashing competitors.
Chatting between different platforms should be easy and secure,” she wrote.

On Monday, Beeper announced on its blog that it has updated the Beeper Mini app to fix the issue that “caused messages not to be sent or received.” It hasn’t
said exactly how it’s made that happen, but notes that the security of Beeper Mini remains unchanged — promising it still includes local, end-to-end encryption
as previously described. Migicovsky told TechCrunch he can’t share anything on that topic beyond what’s already in the blog post.

Though Beeper Mini is running again, there is a caveat: phone number registration isn’t yet working. That means users will have to sign in with an Apple ID.
</item 7>

<item 8>
Beeper Mini is back in operation after Apple’s attempt to shut it down (TechCrunch, 2023-12-11T18:08:48+00:00)
Beeper Mini, the app bringing blue bubble iMessage texts to Android users, is back in operation on Monday after a long weekend that saw Apple putting an end to
Beeper’s services, claiming it was a security risk. Now, the startup behind the new app says it has resumed functioning, but hasn’t disclosed how that’s been
made possible. In addition, Beeper Mini will be free for the time being given the unstable nature of Beeper’s status and Apple’s attempts at shutting it down.

Founded by former Pebble smartwatch founder Eric Migicovsky, Beeper had originally been working on an app that aggregated all of a user’s chat apps into a
single interface. However, upon the discovery of new technology that allowed it to reverse-engineer the iMessage protocol, it set out to build a new app called
Beeper Mini that solely focused on bringing iMessage chats to Android users for $1.99 per month. That included support for high-quality photos and videos,
tapback reactions, typing indicators, read receipts and more.
</item 8>

<item 9>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
That means that Beeper Mini can function as a true iMessage client, supporting high-resolution photos and videos, threads, replies, read receipts, direct
messages and group chats, tapback emoji reactions, editing and unsending messages, as well as support for stickers, GIFs, voice notes and more. Not yet
supported are features like live location sharing, message effects and support for FaceTime audio and video calls. Users will not need an Apple ID to use Beeper
Mini.

“It’s a full-blown iMessage app,” Migicovsky says. “For all intents and purposes, Beeper Mini looks like an iPhone [sent it].”

Apple, then, may not be able to simply block Beeper Mini texts automatically if it wanted to take action against the company, though the co-founder admits how
Apple will react is still uncertain.

However, he points to a provision in copyright law, The Digital Millennium Copyright Act (DMCA 1201 F), which says that reverse-engineering for the purposes of
interoperability is protected.
</item 9>

<item 10>
Beeper reversed engineered iMessage to bring blue bubble texts to Android users (TechCrunch, 2023-12-05T15:00:57+00:00)
And when you send a message, you’re actually sending a message to the Mac Mini, which then forwards it to iMessage,” he explains. “Beeper Mini is a native
implementation of the iMessage protocol.”

Beeper does not have access to the contents of users’ messages, the company claims. And unlike the recently paused efforts by Sunbird, which had been trying to
solve the same problem, messages are not sent in clear text.

Instead, the message you send from an Android phone using Beeper Mini is end-to-end encrypted to the recipient, the startup says. It’s encrypted on the device
before it leaves the app. Encryption keys are exclusively stored on your phone within the Android filesystem, similar to other apps like Signal and WhatsApp.
The app doesn’t connect to any servers at Beeper itself, only to Apple servers, the way a “real” iMessage text would.
</item 10>
=== END ITEM id=07d3a847a618098c ===

=== ITEM id=6765d827dd625521 ===
Gold sentences and facts:
Sentence 0: Rangers also make a lot of choices during character creation (and leveling up) that determine how they play.
  Fact 0.0: Rangers make a lot of choices during character creation.
  Fact 0.1: Rangers make a lot of choices during leveling up.
  Fact 0.2: The choices Rangers make during character creation and leveling up determine how they play.
Sentence 1: During character creation, you must pick the otherworldy patron that you’ve decided to serve.
  Fact 1.0: During character creation, you must pick an otherworldly patron.
  Fact 1.1: The patron you must pick during character creation is the one you have decided to serve.

Retrieved items:
<item 1>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
They also get a way to heal themselves and way to get a second attack once per short rest.

At level 3, you get to choose a subclass from Battle Master, Eldritch Knight, and Champion. Battlemasters get a set of Superiority Dice that let you perform
Maneuvers to control the flow of battle and the battlefield. Eldritch Knights add magic into the Fighter’s arsenal. Champions keep it simple with a better
change to deal a Critical Hit (for double damage).

Champion is the best subclass for a Fighter. Consult our Fighter build guide for more details on why, and to see which skills and feats to pick.

Best ranged class for beginners: Ranger

Rangers are agile hunters that focus on Dexterity as their primary Ability. Rangers are an interesting mix of melee, stealth, skills, and magic. That also makes
it harder to get the most out of the class. Rangers get Light and Medium Armors and some of the more basic weapons.
</item 1>

<item 2>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Picking the best class in Baldurs Gate 3 is one of the most important decisions you’ll make in the game — and you have to do it right at the start, with minimal
context to work off.

Your class mostly determines how your character will fight in Baldur’s Gate, but with 12 classes — and 46 subclasses total across them all — it’s not really an
easy decision. Here’s what to know about finding the best class (and the best subclass) for you in Baldur’s Gate 3.

All BG3 classes are ‘good’

The classes in Baldur’s Gate 3 — and in pen-and-paper D&D, for that matter — are all fairly well-balanced. There’s no single best class (barrelmancer aside) in
Baldur’s Gate 3, and the game gives you as many opportunities for creative problem solving as it does for punching. Finding the best class really just boils
down to how you want to play.
</item 2>

<item 3>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Rangers get Light and Medium Armors and some of the more basic weapons.

Rangers also make a lot of choices during character creation (and leveling up) that determine how they play. At level 1, you’ll choose a Favored Enemy — a
preferred quarry — and a preferred environment that grants you some magical spells.

At level 3, you get to choose a Ranger subclass from Beast Master, Gloom Stalker, or Hunter. Beast Masters get a Companion animal that fights along side you,
Gloom Stalkers focus on stealth, and Hunters get to specialize their melee abilities.

Hunter is the best subclass for a Ranger. Consult our Ranger build guide for more details on why, and to see which skills and feats to pick.

Best class for damage: Barbarian

Barbarians are all about combat. They get light and medium armor (and shields), but they’ve also got the option to keep their AC up even without wearing armor.
</item 3>

<item 4>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
For more information on the classes in Baldur’s Gate 3, consult our overviews on the Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue,
Sorcerer, Warlock, and Wizard classes.
</item 4>

<item 5>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
If you’re bound to The Archfey, you’ll gain an action named “Fey Presence,” which will charm or frighten nearby foes.

The Archfey also has two exclusive spells — Faerie Fire and Sleep.

Best Warlock subclass and build in BG3

Baldur’s Gate 3 races don’t really have any direct impact on your class, but certain races and subraces come with features that play well with certain classes
like extra movement or proficiency with certain weapons.
</item 5>

<item 6>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
For weapons, Barbarians get a wide range of heavy hitters like battleaxes and warhammers. Every Barbarian gets the Rage ability that lets them hit even harder
and reduce damage during combat (and only during combat).

When you hit level 3, you get to choose from the Wildheart, Berserker, or Wild Magic subclasses. Wildheart lets you modify your Rage ability with animal-themed
tweaks like healing, jumping, or a stampede. Berserker is a lot more straightforward and focuses on hitting (extra) things and throwing. And Wild Magic adds a
magical flair to your Rage with a random magical effect and improved saving throws against magic for you and your allies.

Berserker is the best subclass for a Barbarian. Consult our Barbarian build guide for more details on why, and to see which skills and feats to pick.

Best class for magic: Druid

Druids are the nature-based casters of Baldur’s Gate 3.
</item 6>

<item 7>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
The Warlock is one of twelve classes in Baldur’s Gate 3. Warlocks are extreme knowledge-seekers who are willing to form a pact with otherwordly beings in order
to gain ancient secrets that’ll bolster their magical arsenal.

To deal with such powerful creatures, Warlocks need to have strong mental power. They need to be able to make a deal with the devil and survive. As such, their
primary ability score is Charisma.

There’s no single best class in Baldur’s Gate 3, and the game gives you as many opportunities for creative problem solving as it does for punching. Finding the
best class really just boils down to how you want to play. Even when playing solo, you’ll still manage the rest of your party, so you’ll get to experiment with
multiple classes at once.

Read on to find out everything you need to know about the Warlock class in Baldur’s Gate 3, and learn all about the best Warlock subclass, feats, and build.
</item 7>

<item 8>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Finding the best class really just boils down to how you want to play. Even when playing solo, you’ll still manage the rest of your party, so you’ll get to
experiment with multiple classes at once.

BG3 classes have strengths outside of combat

Your character’s class is mostly about combat, but the underlying Ability Score that it relies on affects the other skills you’ll use to interact with the
world. Baldur’s Gate 3 assigns your Ability Scores automatically (though you can tweak it in the character creator), so each class’s primary ability will be
your highest score.

Bards and Sorcerers are both Charisma-based classes, for example. Their high Charisma will also make them good at skills like Persuasion, Deception, and
Intimidation. Out of combat, those classes will be better at talking their way out of (and into) trouble and lying. Wisdom-based classes like Clerics or Druids
will be more insightful and perceptive of their surroundings.
</item 8>

<item 9>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Should you play a custom or an Origin character in BG3?

Baldur’s Gate 3 gives you the option to play as an Origin character, meaning you can just choose one of the NPCs from your party and play as them right out of
the gate. It’s a perfectly fine way to play if you really don’t want to go through the process of picking out a race and background for your character.

That said, you’ll also be limited to that character’s predetermined story. It’s a bit more work at the beginning, but you’ll have a lot more freedom to
experience the entire story of Baldur’s Gate 3 if you make a custom character. With all that in mind...

Which BG3 class (and subclass) is best for you?

Distinctions like caster vs. puncher or melee vs. ranged don’t really work for Baldur’s Gate 3 (or D&D). There’s a lot of overlap in the classes — especially as
you level up.
</item 9>

<item 10>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
For Warlocks, it’s best to focus on weapon and armor proficiencies:

Drow ’s Drow Weapon Training adds weapons like the rapiers and hand crossbows

’s Drow Weapon Training adds weapons like the rapiers and hand crossbows Humans ’ and Half-Elves ’ Civil Militia add spears, pikes, halberds, and glaives along
with shields

’ and ’ Civil Militia add spears, pikes, halberds, and glaives along with shields Githyanki Martial Prodigy means you get medium armor along with longswords and
greatswords

Martial Prodigy means you get medium armor along with longswords and greatswords Dwarven Combat Training gets you battleaxes, handaxes, light hammers, and
warhammers. The Shield Dwarf subrace also gets you Medium Armor

When you’re creating and leveling up your Warlock, Charisma (for spellcasting) is your most important stat.
</item 10>
=== END ITEM id=6765d827dd625521 ===

=== ITEM id=e6ed70b749593d73 ===
Gold sentences and facts:
Sentence 0: As such, they’re never going to replace the need for reducing emissions of CO 2 and other greenhouse gases.
  Fact 0.0: The subject referred to as 'they' is never going to replace the need for reducing emissions of CO2 and other greenhouse gases.
Sentence 1: He stopped flying, became a vegetarian and ditched gasoline-powered cars (he drives a Tesla), cutting his personal emissions by about 90%, according to his math.
  Fact 1.0: He stopped flying.
  Fact 1.1: He became a vegetarian.
  Fact 1.2: He ditched gasoline-powered cars.
  Fact 1.3: He drives a Tesla.
  Fact 1.4: He cut his personal emissions by about 90%, according to his math.

Retrieved items:
<item 1>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Twice, he’s been arrested during climate protests.

His social media posts often evoke heartbreak.

“We’re in the process of losing basically everything,” he wrote recently on X.

Is anyone listening?

climate scientist natural farming (Evan Bush / NBC News)

Not his suburban neighbors in North Carolina, who continue to drive emissions-belching SUVs to big-box stores. Perhaps not even some in the science community,
like those who kicked him out of the American Geophysical Union’s annual fall meeting last year after he unfurled a banner and protested on the major science
conference’s biggest stage.

He has grown increasingly frustrated with President Joe Biden, who signed the Inflation Reduction Act as his signature climate bill. Kalmus thinks it does too
little to shut out the fossil fuel industry.

(Rhodium Group, a nonpartisan think tank, estimates greenhouse gas emissions will drop 32%-42% below 2005 levels by 2030, well short of Biden’s own benchmark
for progress. The inflation bill is responsible for about a quarter of that projected decrease.)
</item 1>

<item 2>
Graph query row:
article: A scientist reckons with climate grief
published_on: 2023-12-24T13:06:02+00:00
person: Peter Kalmus
person_relationship: MENTIONS
publisher: Yahoo News
publisher_relationship: PUBLISHED_BY
topic: climate anxiety
page: A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
</item 2>

<item 3>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
He’s trying to build a gift economy in Belfast, in which neighbors share resources and services with no expectation of repayment. “I’m trying to role-model
post-petrol,” Hughes said. (Evan Bush / NBC News)

Kalmus, now a married father of two teenagers, is introspective, earnest and narrowly focused on climate change in both his personal and professional life. Tall
with an athletic build, a neat beard and a tangle of messy hair that has a skunklike streak of white, he can sometimes come off like an overgrown teenager: He’s
the smartest person in the room, brimming with idealism and still indignant — with a zeal many adults can no longer muster — about how life’s most annoying
systems work (like the U.S. medical system, a symptom of what he calls the “capitalist death cult”).

The changes he made weren’t enough. Kalmus wrote a book on climate change and chained himself to a Chase bank in protest of fossil fuel financing. Twice, he’s
been arrested during climate protests.
</item 3>

<item 4>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
BELFAST, Maine — How does it feel to be a climate scientist in 2023?

Let’s set the scene. Hundreds of picnickers are sprawled across a lawn speckled with colorful blankets. Yachts bob in the bay. Guitars blare, children wrestle
and dancers twirl across a concert lawn in the golden glow of a dying summer night.

It’s a picture-perfect August evening for the seaside town of Belfast’s “Summer Nights” concert series.

Off to the side, Peter Kalmus sits alone. His eyes are closed. His toes are tucked beneath his body. He’s catching an occasional side-eye from onlookers.

“When I’m meditating, I don’t feel anxious,” Kalmus later says, explaining that he strives for two hours of meditation a day to keep his climate anxiety at bay.
Otherwise, “it’s completely overwhelming.”

Many of us avoid thinking too deeply about the worst scenarios of climate change. He is fixated on them.
</item 4>

<item 5>
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
</item 5>

<item 6>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
He is fixated on them.

Kalmus, 49, thinks civilization is on the path to break down, the Biden administration is clueless on climate, and that he might get fired from his job at NASA
if he is arrested for a third time protesting what he views as downright madness: the continued use of fossil fuels.

Kalmus visited the seaside town of Belfast, Maine, in July and August to visit the Possibility Alliance homestead as he looked for inspiration on how to best
live with the effects climate change. (Evan Bush / NBC News)

“It’s kind of weird and lonely,” he said. “People don’t want to talk about this stuff at parties.”

Kalmus, whose strident tweets reach some 340,000 followers on X, is at the vanguard of a group of scientists who are increasingly disillusioned and growing
convinced that more radical action is needed to wake up political leaders.
</item 6>

<item 7>
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
</item 7>

<item 8>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Months later, after returning “kicking and screaming” to the “quote-unquote real world,” Kalmus was planning a return trip to the Possibility Alliance over the
winter holidays.

He was still mulling why the Possibility Alliance’s “low energy” lifestyle had felt so meaningful and what was missing from modern life.

Ethan Hughes his goats out to

“Everything is too fast,” Kalmus said. “It’s obvious to a lot of people, but it’s a new revelation to me. There’s nothing more important in life than
relationships.”

This article was originally published on NBCNews.com
</item 8>

<item 9>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
These materials are harder to make than MXenes. So most research on them has been done with math or computer models. But MBenes could have unique mechanical and
electrical traits, Ozkan’s team finds. And those just might help them last through many, many cycles of CO 2 capture and release.

Each year, human activities spew tens of billions of tons of CO 2 into the air. That’s equivalent to the weight of tens of billions of elephants. Carbon
absorbers, including MXenes or MBenes, can’t catch it all. As such, they’re never going to replace the need for reducing emissions of CO 2 and other greenhouse
gases.

Still, given the urgency of climate change, even a quick or partial fix — such as carbon capture — can’t be ignored. “At this point in time,” says Persson, “we
need to look at all the possible solutions.” And MXene-like materials are emerging as potentially important candidates.
</item 9>

<item 10>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Kalmus, trailed by a reporter, buzzed around the homestead like a giddy kid, hiking back and forth to a home in an eco-village near the Possibility Alliance
where he was living and working on his research. He tramped down a forest trail in bare feet, paused work for a midday dip in a shady stream and never changed
his T-shirt.

He ground his coffee with a mortar and pestle, thumbed through a book of Enneagram personality types on the porch with his son Zane and listened to Ethan’s
stories about sailing across the Atlantic in a cargo ship to avoid the carbon emissions of flying.

“I feel met by Peter,” Sarah Wilcox-Hughes said of Kalmus. “I don’t feel like there are many people prioritizing the needs of the planet as he is and we are.”

Kalmus found life in Belfast, and at the Possibility Alliance, slower, richer and more connected to the natural world.
</item 10>
=== END ITEM id=e6ed70b749593d73 ===

=== ITEM id=352c958cf218e193 ===
Gold sentences and facts:
Sentence 0: Rangers also make a lot of choices during character creation (and leveling up) that determine how they play.
  Fact 0.0: Rangers make a lot of choices during character creation.
  Fact 0.1: Rangers make a lot of choices during leveling up.
  Fact 0.2: The choices Rangers make during character creation and leveling up determine how they play.
Sentence 1: During character creation, you must pick the otherworldy patron that you’ve decided to serve.
  Fact 1.0: During character creation, you must pick an otherworldly patron.
  Fact 1.1: The patron you must pick during character creation is the one you have decided to serve.

Retrieved items:
<item 1>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
For more information on the classes in Baldur’s Gate 3, consult our overviews on the Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue,
Sorcerer, Warlock, and Wizard classes.
</item 1>

<item 2>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Rangers get Light and Medium Armors and some of the more basic weapons.

Rangers also make a lot of choices during character creation (and leveling up) that determine how they play. At level 1, you’ll choose a Favored Enemy — a
preferred quarry — and a preferred environment that grants you some magical spells.

At level 3, you get to choose a Ranger subclass from Beast Master, Gloom Stalker, or Hunter. Beast Masters get a Companion animal that fights along side you,
Gloom Stalkers focus on stealth, and Hunters get to specialize their melee abilities.

Hunter is the best subclass for a Ranger. Consult our Ranger build guide for more details on why, and to see which skills and feats to pick.

Best class for damage: Barbarian

Barbarians are all about combat. They get light and medium armor (and shields), but they’ve also got the option to keep their AC up even without wearing armor.
</item 2>

<item 3>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Should you play a custom or an Origin character in BG3?

Baldur’s Gate 3 gives you the option to play as an Origin character, meaning you can just choose one of the NPCs from your party and play as them right out of
the gate. It’s a perfectly fine way to play if you really don’t want to go through the process of picking out a race and background for your character.

That said, you’ll also be limited to that character’s predetermined story. It’s a bit more work at the beginning, but you’ll have a lot more freedom to
experience the entire story of Baldur’s Gate 3 if you make a custom character. With all that in mind...

Which BG3 class (and subclass) is best for you?

Distinctions like caster vs. puncher or melee vs. ranged don’t really work for Baldur’s Gate 3 (or D&D). There’s a lot of overlap in the classes — especially as
you level up.
</item 3>

<item 4>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
(The Great Old One exclusive) Expeditious Retreat Gain Dash immediately and as a bonus action on each of your turns until this spell ends. Faerie Fire All
targets within the light turn visible, and Attack Rolls against them have advantage. (The Archfey exclusive) Hellish Rebuke React to your next attacker with
flames that deal 2-20 Fire damage. Hex Make your attacks deal an addition 1-6 Necrotic damage to the target and give it Disadvantage on an Ability of your
choosing. Protection from Evil and Good Protect an ally against the attacks and powers of aberrations, celestials, elementals, fey, fiends, and undead. Tasha's
Hideous Laughter Leave a creature Prone with laughter, without the ability to get up. (The Great Old One exclusive) Sleep Put creatures into a magical slumber.
Select targets up to a combined 24 hit points. (The Archfey exclusive) Witch Bolt Link yourself to a target with a bolt of lightning.
</item 4>

<item 5>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Picking the best class in Baldurs Gate 3 is one of the most important decisions you’ll make in the game — and you have to do it right at the start, with minimal
context to work off.

Your class mostly determines how your character will fight in Baldur’s Gate, but with 12 classes — and 46 subclasses total across them all — it’s not really an
easy decision. Here’s what to know about finding the best class (and the best subclass) for you in Baldur’s Gate 3.

All BG3 classes are ‘good’

The classes in Baldur’s Gate 3 — and in pen-and-paper D&D, for that matter — are all fairly well-balanced. There’s no single best class (barrelmancer aside) in
Baldur’s Gate 3, and the game gives you as many opportunities for creative problem solving as it does for punching. Finding the best class really just boils
down to how you want to play.
</item 5>

<item 6>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Wizards don’t get any armor and are limited to just daggers and quarterstaffs (quarterstaves?) in combat, so they’re going to rely on magic to do any damage.

Your Wizard subclass is chosen when you create the character. You’ll choose from eight schools of magic: Abjuration, Conjuration, Divination, Enchantment,
Evocation, Illusion, Necromancy, and Transmutation. Each subclass gets an extra ability themed to the school of magic. Broadly speaking, Abjuration is about
protection and defense, Conjuration is about creating something from nothing, Divination gets to foresee (and change) rolls, Enchantment is about charming and
controlling minds, Evocation shapes and controls spells, Illusion is about trickery, Necromancy controls life and death, and Transmutation focuses on alchemy
and changing the world around you.

Evocation is the best subclass for a Wizard. Consult our Wizard build guide for more details on why, and to see which skills and feats to pick.
</item 6>

<item 7>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
For Warlocks, it’s best to focus on weapon and armor proficiencies:

Drow ’s Drow Weapon Training adds weapons like the rapiers and hand crossbows

’s Drow Weapon Training adds weapons like the rapiers and hand crossbows Humans ’ and Half-Elves ’ Civil Militia add spears, pikes, halberds, and glaives along
with shields

’ and ’ Civil Militia add spears, pikes, halberds, and glaives along with shields Githyanki Martial Prodigy means you get medium armor along with longswords and
greatswords

Martial Prodigy means you get medium armor along with longswords and greatswords Dwarven Combat Training gets you battleaxes, handaxes, light hammers, and
warhammers. The Shield Dwarf subrace also gets you Medium Armor

When you’re creating and leveling up your Warlock, Charisma (for spellcasting) is your most important stat.
</item 7>

<item 8>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
They also get a way to heal themselves and way to get a second attack once per short rest.

At level 3, you get to choose a subclass from Battle Master, Eldritch Knight, and Champion. Battlemasters get a set of Superiority Dice that let you perform
Maneuvers to control the flow of battle and the battlefield. Eldritch Knights add magic into the Fighter’s arsenal. Champions keep it simple with a better
change to deal a Critical Hit (for double damage).

Champion is the best subclass for a Fighter. Consult our Fighter build guide for more details on why, and to see which skills and feats to pick.

Best ranged class for beginners: Ranger

Rangers are agile hunters that focus on Dexterity as their primary Ability. Rangers are an interesting mix of melee, stealth, skills, and magic. That also makes
it harder to get the most out of the class. Rangers get Light and Medium Armors and some of the more basic weapons.
</item 8>

<item 9>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Finding the best class really just boils down to how you want to play. Even when playing solo, you’ll still manage the rest of your party, so you’ll get to
experiment with multiple classes at once.

BG3 classes have strengths outside of combat

Your character’s class is mostly about combat, but the underlying Ability Score that it relies on affects the other skills you’ll use to interact with the
world. Baldur’s Gate 3 assigns your Ability Scores automatically (though you can tweak it in the character creator), so each class’s primary ability will be
your highest score.

Bards and Sorcerers are both Charisma-based classes, for example. Their high Charisma will also make them good at skills like Persuasion, Deception, and
Intimidation. Out of combat, those classes will be better at talking their way out of (and into) trouble and lying. Wisdom-based classes like Clerics or Druids
will be more insightful and perceptive of their surroundings.
</item 9>

<item 10>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Spell Sniper. You learn a cantrip, and the number you need to roll a Critical Hit while attacking is reduced by 1. This effect can stack.

For more information on the classes in Baldur’s Gate 3, consult our overviews on the Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue,
Sorcerer, and Wizard classes. If you’re stumped by the choice, consult our guide on what class to start with.
</item 10>
=== END ITEM id=352c958cf218e193 ===

=== ITEM id=b80a75e6bd55f208 ===
Gold sentences and facts:
Sentence 0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 0.0: Wang testified that he heard Bankman-Fried tell another trader.
  Fact 0.1: Bankman-Fried said it was okay to continue withdrawing money from the account.
  Fact 0.2: The withdrawals were okay as long as the amount was less than FTX's total trading revenue.
Sentence 1: Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry.
  Fact 1.0: Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto.
  Fact 1.1: Bankman-Fried made himself out to be the trustworthy face of a sometimes-shady industry.
Sentence 2: It was not until Judge Lewis Kaplan intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance, that Bankman-Fried admitted he’d been told.
  Fact 2.0: Judge Lewis Kaplan intervened to ask if Bankman-Fried had ever been told by Yedidia about that money, in words or in substance.
  Fact 2.1: Bankman-Fried admitted he had been told about that money.
  Fact 2.2: Bankman-Fried admitted this only after Judge Lewis Kaplan intervened.
Sentence 3: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 3.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 3.1: The defense countered that the FTX founder acted in good faith.
  Fact 3.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 3.3: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“All of that was built on lies,” Rehn said.

In his opening statement, Rehn dodged explaining cryptocurrency to the jury. Instead, he punched hard on Bankman-Fried lying and stealing.

Bankman-Fried sat almost motionless, occasionally glancing at Rehn, as the prosecutor told the jury that Bankman-Fried sold stock in FTX and borrowed millions
from lenders by lying.

The story Rehn told is familiar to anyone following the news. In May and June of 2022, Alameda Research — the crypto trading company ostensibly helmed by
Caroline Ellison — didn’t have enough to pay its bills, so it pulled customer money to repay loans. By September, the hole in the FTX balance sheet was so big
that customers could never be repaid.

FTX “didn’t have a chief risk officer, which became an issue when the storm hit.”

When CoinDesk published its article in November 2022, people realized FTX was a house of cards, Rehn said. Meanwhile, Bankman-Fried tweeted.
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 2>

<item 3>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
It is honestly kind of incredible to watch a man torpedo his own credibility on direct testimony. We’re not even at the cross yet, and the judge has already
instructed him to answer the question he’s being asked by his own lawyer.

The jury is watching all of this intently.

The main thing that’s been clear so far from Bankman-Fried’s testimony is that the man really loves the sound of his own voice. So far, the count of “Objection,
narrative!” to Bankman-Fried’s answers, followed by “Sustained” is at three.

Also, sometimes when Bankman-Fried says “we,” he only means himself.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, has been found guilty on seven counts including charges of wire fraud. FTX was a fraud
“from the start,” the Securities and Exchange Commission alleged — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer
funds.” Follow along for all the latest news and regular updates from the trial.
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.
</item 4>

<item 5>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
That was before he realized FTX had stolen customer funds — at which point, he resigned.

Gary Wang, the co-founder of FTX and Alameda Research and another college roommate. He took the stand after Yedidia and sang like a canary about the secret
privileges he’d created for Alameda in the FTX code.

Nishad Singh, a longtime family friend, who copped to what sounded an awful lot like classic embezzlement. He said he’d backdated transactions on FTX to help
create doctored balance sheets to show to investors. He pleaded guilty to six charges, including wire fraud, money laundering, and conspiracy counts around his
political donations.

Caroline Ellison, Bankman-Fried’s ex-girlfriend. She told us about the seven phony balance sheets she’d created. She was clear: Alameda took customer funds;
Bankman-Fried knew about it and even invested them against her advice.

Joe Bankman, Bankman-Fried’s father.
</item 5>

<item 6>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 6>

<item 7>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.
</item 7>

<item 8>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Whatever records the Southern District of New York has for the transactions will be shown.

“What conversations happened between him and his co-conspirators that are now cooperating against him?”

The second part is where all the drama is likely to come, says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law
firm Withers. Prosecutors have to connect those transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it
anyway.

“What was he saying to his parents and his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are
now cooperating against him?”

To further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”

“We had him going out directly to the internet,” LaVigne says.
</item 8>

<item 9>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=b80a75e6bd55f208 ===

=== ITEM id=c5d10dfe896f94d2 ===
Gold sentences and facts:
Sentence 0: “Certainly there wasn’t a valid alternative to Google at the time,” Cue said.
  Fact 0.0: Cue said there was not a valid alternative to Google at the time.
  Fact 0.1: Cue said 'Certainly there wasn't a valid alternative to Google at the time.'
Sentence 1: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 1.0: The party referred to as 'they' gets other benefits from Android.
  Fact 1.1: Google could solve this problem today if Google wanted to.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 2>

<item 3>
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
</item 3>

<item 4>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 4>

<item 5>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 5>

<item 6>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
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
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
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
=== END ITEM id=c5d10dfe896f94d2 ===
