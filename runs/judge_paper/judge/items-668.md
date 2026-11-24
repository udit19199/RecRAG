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

=== ITEM id=b1023334eaed0eb0 ===
Gold sentences and facts:
Sentence 0: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.0: Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.1: Valve has made big improvements with its new iteration.
  Fact 0.2: The minor physical upgrades are part of the new iteration alongside the big improvements.
Sentence 1: We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.
  Fact 1.0: Valve looks at a store that could be both.
  Fact 1.1: Valve decided to focus exclusively on games.
Sentence 2: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 2.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 2.1: Valve has units ready to ship on November 16th.

Retrieved items:
<item 1>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
As we donned Samus’ suit and explored strange planets, aggressive alien lifeforms could now get right in our faces, forcing us to dodge, strafe, and roll (in
morph ball form, naturally) using all three dimensions. No longer would we sit back and watch as Samus dipped her toe into a pool of lava; in first-person, as
molten fire spread over our visor, we’d really feel the pressure to find that Varia Suit upgrade. And perhaps most importantly, from behind Samus’ visor, we
gained the ability to scan our enemies and environment, collecting and translating logs from the long-dead Chozo aliens who once inhabited these now-hostile
places.

The world of Prime is harsh and unrelenting. (Save points will, at times, be quite far from one another.) But it’s worth buckling down and pushing through the
pain points to discover this world’s secrets. —M. Myers

18.
</item 1>

<item 2>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Currently, more than 10,000 titles on Steam fall under the playable or verified labels, with more added every day.

When will the Steam Deck OLED be available?

The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day. The company has lowered the price of its LCD
lineup and will sell through its backstock of 64GB and 512GB models until they’re all gone. It’ll continue selling the 256GB LCD model at a reduced price of
$400, while the 512GB OLED model will cost $550 and the 1TB OLED model will cost $650. There’s also a limited edition 1TB OLED version priced at $680 that comes
with a special carrying case, a translucent gray body and orange accents.

Valve

Wrap-up

According to the developers I spoke with at Valve, this is the definitive version of the Steam Deck.
</item 2>

<item 3>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
(Parts will once again be sold through iFixit.) However, I did enjoy the new power supply, with its faster charging (45 minutes gets you from 20% to 80%) and
2.5-meter cable that stretches the length of my couch.

One small, charming detail: The carrying case now has a tinier carrying case within it. If you want to be extra cautious, stick to the original bulky case. But
if you need to toss the Steam Deck in your backpack for a day trip? Go with the slim case that has funny little thumbstick nipples!

All of the tweaks, large and small, amount to a perfect handheld — made better by the price, which remains the same as the original Steam Deck available today,
just that the OLED will come with more stuff.
</item 3>

<item 4>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The improved 90 Hz refresh rate falls short of the Asus ROG Ally’s 120 Hz, though it’s worth considering that the higher the frame rate, the faster the battery
drain — assuming you can even spot the difference between 90 Hz and 120 Hz.

You know what’s more interesting than a list of technical specs? Actually seeing the screen. It’s gorgeous. On Sunday morning, I picked at Sega’s Like a Dragon
Gaiden: The Man Who Erased His Name while watching football and Christmas movies with my family. The new Yakuza game looks fantastic on high settings at 60 fps,
its night sequences flaunting the OLED’s dark blacks and the abundance of neon signage showcasing the HDR.

I got so distracted pulverizing bad dudes that it took me some time to notice the Steam Deck OLED’s most welcome feat. When playing AAA games on the original
Steam Deck, I had always kept the handheld plugged into an outlet. But with the OLED model, a couple of hours had passed before I realized I’d forgotten to
connect the charger.
</item 4>

<item 5>
The best portable chargers for your gear (Polygon, 2023-11-17T13:10:12+00:00)
You can get this two-pack of 100 W-ready, six-foot-long, USB-C-to-C cables at Amazon for $15.99.

Related The best Nintendo Switch accessories

Best power bank for your Steam Deck

Each Steam Deck console includes a 45 W USB-C wall adapter, so it’s a good idea to find a power bank that can meet that power level when you’re on the go.
Thankfully, some options exist that can do this. One thing to note during your search is that only batteries that support the PD 3.0 spec, including the two
options below, can recharge the Steam Deck at its fastest speed.

Related The best Steam Deck accessories

Best portable charger for your laptop

The number of laptops that can recharge via USB-C is growing, making it possible to use the same charger to refill your laptop as you do your phone or Nintendo
Switch. For instance, Apple’s entire MacBook lineup allows it, as do several popular Windows laptops, like the Dell XPS 13, Razer’s Blade gaming laptop, and
more.
</item 5>

<item 6>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Viewfinder

Developer: Sad Owl Studios

Where to play: PlayStation 5 and Windows PC

At the heart of Viewfinder is a magic trick that never gets old: You take a photo of the environment and paste that perspective into the game world to create a
bridge or reveal a needed trinket. The experience of seeing reality distorted by your hands so easily is nearly on the level of thinking with Portals for the
first time. You’ll ponder a solution for minutes, sure it’s impossible and that the developers must have made a mistake, before you’re struck by a eureka moment
like a lightning bolt.

The game’s lesson on perspective extends to its narrative, which, much like the photos you use to manipulate the world, contains multitudes. Though it’s told in
fairly typical video game-y audio monologues that you uncover from the game’s trippy environments, the journey is powerful and topical. Just as differing
perspectives can allow for unique solutions, they can also obscure truths that lead us to deny the reality that’s right in front of us.
</item 6>

<item 7>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
These days, Steam generally peaks at over 30 million, with valleys of 20 million. Fortnite’s valleys are closer to 3.8 million.]

Tim Sweeney says he plays Jellie in Fortnite. Image: Epic Games

Okay, let’s do a lightning round. What’s your favorite Fortnite skin?

I play as Jellie! Jellyfish. You know, the big tentacles? It’s just so cool!

Will Epic ever announce Unreal Tournament 3 X?

Forever in motion, the future is.

Why is Fortnite still not playable on Steam Deck?

If we only had a few more programmers. It’s the Linux problem. I love the Steam Deck hardware. Valve has done an amazing job there; I wish they would get to
tens of millions of users, at which point it would actually make sense to support it.
</item 7>

<item 8>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
It’s less top-heavy, and it does technically weigh less. These small adjustments have made a noticeable difference during my playtime, and the device feels
slightly more manageable as a handheld — or maybe I’m just too distracted by all of its bright, pretty colors to care about the cramping in my palms.

The thumbsticks on the new handheld have an improved texture that shouldn’t turn gray after a few months’ use, and they feel perfectly serviceable. In my
hands-on time so far, I've found that the OLED touchscreen responds immediately to input, and the haptic feedback on the built-in trackpads feels even more
precise, populating in reactive pin-pricks under the textured plastic.

I spent a few cozy nights on my couch swapping between the LCD and OLED Steam Decks, and honestly, after I’m done with this review, I’ll probably retire my
original model. The Steam Deck OLED is sharper and more responsive; indie games, AAA games and plain old Steam menus look so much better on the new hardware.
</item 8>

<item 9>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Manufacturers typically use die-shrinks to either improve performance or efficiency, and Valve has chosen the latter. The max frequency of both the GPU and CPU
are identical between the LCD and OLED models, so for developers tuning their games for Steam Decks, they only have a single performance target.

In concert with the more-efficient processor, there’s also a 25 percent larger battery with a 50Whr capacity, compared to the original’s 40Whr. According to
Valve, these changes combined result in a 30-50 percent increase in longevity between the two generations — the official battery life claim is "three to 12
hours." In our review, we measured a playtime increase of 62 percent when playing Hades on the OLED model versus our 18-month-old original. (Batteries degrade
over time, which explains why our gains were more pronounced.)

The new internals make the Steam Deck OLED marginally lighter, and also cooler to the touch when playing games, aided by a slightly larger fan.
</item 9>

<item 10>
Save on the winners from the 2023 Game Awards this weekend (Polygon, 2023-12-09T15:00:00+00:00)
We’re coming up for air from our coverage of The Game Awards to loop our readers in on some of the best deals happening this weekend. If you’re looking to
experience some of the winners from the 2023 Game Awards for yourself, you can find discounts on a variety of titles on Steam. Some other highlights you might
want to check out include a sitewide sale on Blu-Rays from The Criterion Collection, and the Xbox Series X matching its Black Friday price on Walmart and
Amazon.

As we do every weekend here at Polygon, we’ve rounded up a collection of our favorite deals from the worlds of gaming and entertainment, along with some of the
best-selling products that have made a recent appearance on our site.

The best gaming deals this week

Steam is discounting many contenders from the 2023 Game Awards and winners from the past nine years.
</item 10>

<item 11>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The original Steam Deck had a 7-inch 800p LCD running at 60Hz. The new model squeezes a 7.4-inch OLED screen into the same space, significantly shrinking the
bezels around the display. The new panel runs at the same resolution, but with a refresh rate of up to 90Hz. It’s also HDR, covering 110 percent of the P3 color
space with a peak brightness of 1,000 nits — the old model maxed out at around 400 nits. Like all OLED screens, it has pure blacks and near-instantaneous
response times. As before, anti-glare etched glass is available, but only on the premium model.

The Steam Deck OLED has a die-shrunk version of the same custom APU found in the original Steam Deck. The new APU is produced on a 6nm process, rather than the
older 7nm process.
</item 11>

<item 12>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Other improvements include support for both Wi-Fi 6E and Bluetooth 5.3, together with a new antenna array that will give a better experience when using both at
the same time. There’s also a reworked power supply (it has a logo now!) with a longer 2.5m cable that will charge the OLED model from 20 to 80 percent in “as
little as 45 minutes.” The thumbsticks have also been made 1mm taller, and the touch pads' haptics are slightly sharper.

Valve

The Steam Deck OLED starts at $549, and will replace all but the entry-level model in the range.
</item 12>

<item 13>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
GPU updates were the centerpiece of the news for several reasons. One is a feature the company calls Dynamic Caching. Apple writes,

With Dynamic Caching, only the exact amount of memory needed is used for each task. This is an industry first, transparent to developers, and the cornerstone of
the new GPU architecture. It dramatically increases the average utilization of the GPU, which significantly increases performance for the most demanding pro
apps and games.

More simply put, the idea is allocating just the right amount of local memory to a given task. It’s something Apple has been working on for a while. The other
two additions should be more familiar for anyone who’s spent time behind a game controller. The first is ray tracing, a feature designed to bring added realism
to the gaming experience, though more accurate depictions of lighting and reflections.

That feature is now hardware-accelerated, along with mesh shading. The triangle computer shader makes processing complex geometrical images more efficient.
</item 13>

<item 14>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$400 at Amazon (refurbished)

Jsaux Mod Case for Steam Deck $ 24 $ 30 20 % off $ 24 $ 24 $ 30 20 % off A protective modular case system for Valve’s Steam Deck. It consists of a base case
with grippy texture, a built-in kickstand, a removable lid, and a rear mounting system that can attach a power bank using an adjustable strap or Jsaux’s add-on
cooling fan. $24 at Amazon

Jsaux’s 45W USB-C charger with a five-foot cable and included right-angle adapter may be designed for the Steam Deck, but it’s also a great charger for a
Nintendo Switch or just about any USB-C phone or tablet. It’s just $15.19 ($3 off) right now at Amazon.

with a five-foot cable and included right-angle adapter may be designed for the Steam Deck, but it’s also a great charger for a Nintendo Switch or just about
any USB-C phone or tablet.
</item 14>

<item 15>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The immediate reality is a bit of a mess and a little complicated, but this, Valve says, will be the Steam Deck line-up moving forward:

256GB Steam Deck LCD ($399)

512GB Steam Deck OLED ($549)

1TB Steam Deck OLED ($649)

As a result of the changes, the 64GB LCD Steam Deck (previously sold at $399) will drop to $349, and the 512GB LCD (previously $649) will drop to $499. Once
Valve sells through its inventory of both, those lines will be discontinued. Valve says it will continue to support all LCD models with software updates.

So, the long-term lineup starts at the same $399 price, which gets you the old LCD Steam Deck, but with 256GB of storage instead of 64GB. The new step-up model
costs $20 more at $549, but comes with an OLED display and the improvements outlined above, plus double the storage.
</item 15>

<item 16>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The Steam Deck OLED now includes a Wi-Fi 6E radio (great if you have a modern router and connection to make use of it) and a dedicated Bluetooth module, which
allows for improved audio quality with wireless headphones, the option to wake the Steam Deck from a controller, and the capability to connect many Bluetooth
controllers for multiplayer. The haptics are noticeably improved, with a wider frequency range. And the touchscreen is markedly better — it’s responsive to the
point where I don’t feel the need to plug in a USB-C keyboard anytime I need to type a few words.

A review for spec specifics If you’re the sort of person who likes the nitty-gritty technical details of screens and batteries, we also recommend reading The
Verge’s review.

Valve also claims that the Steam Deck OLED is more repairable, which, mercifully, I didn’t have the need to test for myself. (Parts will once again be sold
through iFixit.)
</item 16>

<item 17>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The gameplay didn’t just help me stick with the game, but instead allowed my excitement to bubble over every time I took on a new mission. —Ana Diaz

15. Cyberpunk 2077: Phantom Liberty

Developer: CD Projekt Red

Where to play: PlayStation 5, Windows PC, and Xbox Series X

Like the best DLC, Phantom Liberty does as much to expand on the base game as it does to reframe it. Alongside the sweeping Patch 2.0, which revamped Cyberpunk
2077’s role-playing progression systems, improved its enemy AI, and course-corrected a litany of other details, Phantom Liberty also adds a whole new district
to the dystopian world of Night City, complete with its own spy-thriller storyline.

The resulting Cyberpunk 2077 is not an entirely different beast than the one that was released in 2020, but it is a more evolved one.
</item 17>

<item 18>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
“I like the PS5 DualSense because it’s incredibly precise — using it, after coming from previous gen controllers, felt like putting on glasses and seeing the
world come into focus” —Nicole Clark

To get around using Steam entirely, you can download an app called DS4Windows along with some drivers, which ensures game compatibility on PC by emulating the
DualSense as an Xbox controller. The app gives you a host of settings to customize the DualSense’s button mapping, if you wish. And, in case you want to go down
the rabbit hole of figuring out which PC games support the DualSense’s haptics and adaptive trigger features (and the steps to getting them to work), the site
PCGamingWiki has you covered.

Related Here are the best PS5 accessories

The DualSense is the official controller for the PS5. In games that support its features, it can have realistic-feeling haptic feedback and gyroscopic motion
controls, and its triggers can provide tension to make certain actions feel more realistic. There’s no better controller for the PS5 than this one.
</item 18>

<item 19>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The triangle computer shader makes processing complex geometrical images more efficient. That, in turn, allows for higher-quality graphics and more scene
detail.

The results are impressive. I’ve been spending a bit of time with the Soulslike title, Lies of P. It’s basically Pinocchio with more swords. It’s of a small –
but increasing – number of day-and-date releases for macOS. A couple of years ago, it would have been inconceivable for the operating system to get the game the
same day as PS5, Xbox Series X/S and Windows. That said, the macOS gaming library still has a long way to go.

Indeed, the gameplay was smooth and detailed – something that also would have seemed impossible not all that long ago. Sony’s PS5 DualShock controller connects
seamlessly and makes for a fun portable gaming experience.

One thing I noticed during last week’s presentation is that Apple is leaning less heavily on Intel comparisons.
</item 19>

<item 20>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—Mike Mahardy

How the Polygon top 50 list works

Over the past few weeks, the Polygon staff voted, championed, debated, and ultimately threw up its hands and marveled at the list of mammoths, curiosities,
puzzle boxes, and black holes that is our top 50 games of 2023. Any video games that were released in 2023, received substantial updates in 2023, or achieved
renewed cultural relevance in 2023 were eligible for this list. Last year, the cutoff for consideration was Nov. 30. (You’ll notice a certain Firaxis Games
joint fairly high up our list.) This year, the cutoff was the same. Should we be thoroughly enamored with Warhammer 40,000: Rogue Trader or Avatar: Frontiers of
Pandora, we’ll make sure they’re considered for next year’s top 50.

Top 50

50. Mr. Sun’s Hatbox

Developer: Kenny Sun

Where to play: Nintendo Switch and Windows PC

Mr.
</item 20>

<item 21>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
When asked about the potential for future hardware — like a return of the Steam Machine or an updated Steam Controller — there were no firm updates, but there
does seem to be something happening. Griffais agreed that enabling the Steam Deck to work as a hybrid living room console was a "side goal" of the team, and
noted that everything Valve does to improve SteamOS will benefit other devices. Coomer added that Valve is still working internally on "a bunch of controllers
and prototypes." If SteamOS is being used on larger displays, the team would want to augment that experience with better input options. "We don't have any
announcements on that front, but it's why we're building prototypes and thinking about the future there," Coomer said.

The Steam Deck OLED will be available to order on November 16th at 1PM ET. Valve expects units to begin shipping as soon as orders start to roll in. The updated
pricing on the existing LCD models is already in effect.
</item 21>

<item 22>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is the highlight of the device, while things like faster charging, improved antennas and smoother haptics are welcome bonuses. More
importantly, this is Valve renewing its commitment to portable PC gaming, and that’s a relief to see. The Steam Deck is the granddaddy of handheld PCs and Valve
has the resources to continue innovating in this space. For now, the OLED model is a half-step toward the Steam Deck 2, which may be the final version of the
device — unless Valve finally learns how to count to three.
</item 22>

<item 23>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Playing the Steam Decks back-to-back highlighted another significant quality-of-life improvement: heat management. My palms have gotten toasty playing the OLED
model, but they haven’t broken a sweat like they often do with the original. The new fan may be larger, but it’s not any louder, and it clearly works a treat.

The Steam Deck OLED has a new Wi-Fi 6E module and improved antennas that should enable faster downloads for anyone with a compatible router. I don’t have a 6E
router or multi-gigabit Wi-Fi, but with a regular Wi-Fi 6 router, the top download speed I saw on the new Steam Deck was 562mbps, which was about 10 percent
faster than the old Steam Deck, and 10 percent slower than the speeds I get on my MacBook Pro. The device uses Bluetooth 5.3 and adds a dedicated antenna, which
should improve functionality when using Wi-Fi and Bluetooth at the same time.
</item 23>

<item 24>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Valve Valve Steam Deck OLED SCORE 89 Engadget 89 Critics - Not yet scored N/A Users - Not yet scored N/A Pros Improved battery life

Bright and smooth visuals

Better thermals

Feels less top-heavy

Long charging cable Cons Still ridiculously large

Seriously it’s so big

Valve Valve Steam Deck OLED This isn't the Steam Deck 2 some are waiting for, it's just a welcome improvement on an already good device. $549 at Steam

OLED screen

The HDR OLED display looks fantastic. It features a wider P3 color gamut, pure blacks and a peak brightness of 1,000 nits, compared with the original max of 400
nits. In action, this means the screen’s bezels meld with the black of any game, creating a more polished frame for retro aspect ratios and providing room for
higher-contrast colors. The screen on the new model doesn’t support variable refresh rates (VRR), but it now tops out at 90Hz, while the original could only hit
60Hz.
</item 24>

<item 25>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The range still tops out at $649, but you’re getting a lot more for your money: All of the hardware improvements, 1TB of storage instead of 512GB and an
improved carry case that can pop out into a slimmer shell that’s smaller and easier to fit in a bag. As mentioned, it also has the same anti-glare etched glass
as the outgoing 512GB LCD model.

Finally, there’s also a Limited Edition version of the 1TB model, with a translucent shell and a customized carry case, which will be sold at $679. We imagine
stocks will not last very long, as it looks pretty rad:

Valve

While there are no concrete updates on future hardware developments, Valve engineer Pierre-Loup Griffais noted that, so long as interest in handheld gaming
remains high, the company has "plans for successive generations of handhelds." For now, Griffais explained, it's important to Valve that developers don't need
to "split their attention" between different performance targets.
</item 25>

<item 26>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.
</item 26>

<item 27>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—Clayton Ashley

33. System Shock

Developer: Nightdive Studios

Where to play: Windows PC

In a year of several stellar remakes and immersive sims, it wouldn’t have been surprising if System Shock showed its age of nearly 30 years. At least, it would
have been understandable if the changes needed to update the game would leave it nigh unrecognizable. That’s why Nightdive Studios’ accomplish is so impressive:
It updated the 1994 classic with a slick coat of modern paint while also preserving what made the game so thrilling in the first place.

The element where that’s most evident is in the game’s look, which captures all of the original System Shock’s garish cyberpunk neons and frightening enemies in
a style that appears like a modern high-fidelity game at a distance, but subtly transforms into retro pixel bitmaps on closer inspection.
</item 27>

<item 28>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Yes. But you might find that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year,
the upgrade is inescapable.

And if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.

The Steam Deck OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has
affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .
</item 28>

<item 29>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is smooth, crisp and bright, like a perfect lemon tart. (Even then, I do not recommend licking the screen.)

Battery life

One of the downsides of the first Steam Deck was its short battery life, which maxed out at eight hours, and that was only under ideal conditions. Valve says
the new Steam Deck battery will last 30 to 50 percent longer, depending on how you’re playing, and it charges faster — the company claims it will go from 20
percent to 80 percent in 45 minutes.

Custom AMD APU

The Steam Deck OLED has a die-shrunk version of the custom AMD APU found in the original Steam Deck — same CPU and GPU, just more efficient. This is the process
Sony takes advantage of to build the PS4 Slim and PS5 Slim, but instead of collapsing the Steam Deck around smaller hardware, Valve chose to add a bigger
battery and larger fan to its existing chassis.
</item 29>

<item 30>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
In the vein of Event Horizon, Sunshine, and Alien, Dead Space was a paragon for sci-fi horror in a confined and claustrophobic setting. Its remake has brought
that same vision to gorgeous new life, bringing quality-of-life changes and underappreciated updates (it has made several previously useless weapons into viable
tools in protagonist Isaac Clarke’s arsenal), making it hard to imagine ever going back to Visceral Games’ phenomenal original. —M. Mahardy

Related The Dead Space remake changes all the right things

13. Octopath Traveler 2

Developers: Square Enix, Acquire

Where to play: Nintendo Switch, PlayStation 4, PlayStation 5, and Windows PC

The first Octopath Traveler was one of those games that was as enjoyable to play as it was painful: enjoyable because so much of it kicked ass, but painful
because so much of it dragged the positive aspects down. In other words, it stood on the precipice of excellence, but couldn’t quite cross the line.
</item 30>

<item 31>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
For the first year in recent memory, scale did not necessitate tradition, and scope did not preclude getting weird. In 2023, nothing was sacred in video games,
and so they felt more vibrant than ever.

Sure, some of the more “focused” games threw us for a pleasant loop: Dredge begins as a lonely fishing sim before transforming into something otherworldly, and
Humanity morphed from a pensive art project into an all-out war. Dave the Diver, similarly, is not so much about being a diver as it is about running a sushi
restaurant, or hunting for alien artifacts, or conversing with said aliens, or — you get the point. Whether you booted up your Steam Deck for a cross-country
flight or hid your Switch off screen during that boring Zoom meeting, the game you returned to was rarely the one you left behind.

This amorphousness (I’m begging our copy editor to let this one slide, because what other “word” could adequately summarize the video games of 2023?)
</item 31>

<item 32>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
For developers, this means the device has the same performance targets as the original and there’s no need to create games specifically for the new hardware. In
the end, the device weighs about 30 grams less than the original model, or the equivalent of five quarters.

Engadget

In use

Which leads us to the bad news for my tiny hand gang. The Steam Deck OLED is the same monstrous size as the original: 11.7 inches long, 4.5 inches tall and
nearly 2 inches thick. The thumbsticks on the new model are actually 1mm taller than the first version. This was one of my main complaints about the Steam Deck
when I first reviewed it: I have smaller-than-average hands, and the Steam Deck looks and feels absolutely ridiculous when I’m holding it. It’s just massive, no
matter how long my manicure gets. However, the OLED model feels more balanced than the original. It’s less top-heavy, and it does technically weigh less.
</item 32>

<item 33>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$59 at Amazon

Storage for consoles and PCs

A massive 1TB SanDisk Extreme microSD card is now $89.99 $79.99 ($13 off) at Amazon. That can store a whole lot of games on a Steam Deck or Nintendo Switch.

Samsung Evo Select microSD card (512GB) $ 30 $ 40 25 % off $ 30 $ 30 $ 40 25 % off This Samsung Evo Select microSD card offers 512GB of extra storage for a
Nintendo Switch, Steam Deck, or Asus ROG Ally at a good price point. $30 at Amazon

Western Digital WD_Black SN850X SSD $ 70 $ 180 61 % off $ 70 $ 70 $ 180 61 % off The WD_Black SN850X is one of the fastest M.2 drives around, with a sequential
read speed of up to 7,300 MB/s and a write speed of up to 6,600 MB/s.
</item 33>

<item 34>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Engadget

Even with improvements to the battery, thermals and weight distribution, the Steam Deck OLED is still more of a Wii U than a Vita. It’s a beast of a handheld,
and it makes the most sense as a portable console to use around the house, rather than a truly travel-ready device. The new model even comes with an extra-long,
2.5m charging cable, designed specifically with couch-based players in mind. I didn’t test out the docking station with the new Steam Deck, but Valve recently
released a SteamOS update that lets players take advantage of HDR and VRR on external displays, which only drives home the Wii U-ness of it all.

As far as software, you know the drill by now. Valve has rolled out labels for Steam games signifying how well they’ll work on the handheld, with four levels:
unknown, unsupported, playable and verified. Playable games generally come with a small caveat, like having to deploy the on-screen keyboard, while verified
games have been fully optimized for portable play.
</item 34>

<item 35>
Microsoft Activision: What does deal mean for gamers? (BBC News - Technology, 2023-10-13T23:01:24+00:00)
And critically, Microsoft has a habit of making its big blockbuster games available to play on Game Pass, for a monthly fee, on the first day they release.

Image source, EPA-EFE Image caption, Bethesda, which is owned by Microsoft, launched its new game Starfield in 2023 - but only on Xbox and PC

It's up to gamers to decide for themselves whether it's a better deal to buy the game or pay to access it.

But it is worth noting this concerned the UK regulator so much it made sure Activision Blizzard games couldn't be streamed exclusively on Xbox for 15 years.

That might be because Microsoft has form here - when the hotly anticipated Starfield came out in September 2023 to critical acclaim, it was available
exclusively on PC and Xbox, and playable on Game Pass from day one.

If games are still on PlayStation, what's in it for Microsoft?

Even with games still able to appear on PlayStation, it's Microsoft who will be winning out.
</item 35>

<item 36>
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
</item 36>

<item 37>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Because while Valve hasn’t upped the horsepower, it has improved practically everything else. The Steam Deck OLED has longer battery life, faster Wi-Fi, and a
stand-alone Bluetooth module. And the screen isn’t just OLED; it’s HDR OLED — the first HDR OLED screen on any gaming portable, with a peak brightness of 1,000
nits for HDR content and 600 nits for SDR (for comparison, the original screen’s peak brightness was 400 nits). And somehow, with all of these additions, the
Steam Deck OLED is lighter.

So, goodbye, old Steam Deck; hello, new Steam Deck.

The Steam Deck OLED delivers on Valve’s original promise

OLED is in the name for a reason.

The OLED panel in the new Steam Deck is lighter, thinner, and more power efficient than its LCD predecessor. It’s also bigger and faster. Though the device has
the same shape as the original Steam Deck, smaller screen bezels allow for a display that’s slightly larger, at 7.4 inches compared to 7 inches.
</item 37>

<item 38>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.
</item 38>

<item 39>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The panel has a wider color gamut (110% DCI-P3, for the visual obsessives) and the beautiful, inky blacks that OLED dorks like myself adore.

Steam Deck OLED’s bigger, better battery Valve’s promise of 30-50% more battery raised one obvious question: Is that comparing the OLED model with a Steam Deck
running its original firmware (which had poor battery life), or a Steam Deck using the latest firmware? Valve’s answer: the latter! So if you’re a Steam Deck
owner, expect those big battery improvements over your current Steam Deck experience.

The OLED panel also has a 90 Hz refresh rate — perfect for frame doubling from 45 frames per second. In case that jargon means nothing to you, suffice it to say
that gameplay on the Steam Deck OLED looks smoother.
</item 39>

<item 40>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The first thing to know about the Steam Deck OLED is that it’s not the Steam Deck 2. The Steam Deck OLED is a mid-cycle refresh from Valve, similar to the
Switch OLED upgrade, but there’s a lot more going on internally here. Its screen is better, its battery life is better, its thumbsticks are better, the haptics
are better, and its chip and thermals are better. One thing that hasn’t changed is its form factor — this is still one of the biggest babies on the handheld
block.

The new Steam Deck’s main upgrade is its OLED screen, which replaces the original LCD and, crucially, has native support for HDR and faster frame rates. The
OLED also has smaller bezels, making the display larger at 7.4 inches, compared with the original 7-inch screen.
</item 40>

<item 41>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to carve out more of it) that lets Valve’s new
handheld truly deliver on its promise of a portable gaming PC.

All the tiny, delightful improvements of Steam Deck OLED

Since the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will continue with
both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s upgrades. The
original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.

Some changes — even smallish ones — demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new
iteration.
</item 41>

<item 42>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Valve will be selling three models:

$649 - 1 TB OLED (up from 512 GB)

$529 - 512 GB OLED (up from 256 GB)

$399 - 256 GB non-OLED Steam Deck (up from 64 GB)

Plus, for a limited time, the company will offer a special edition 1 TB Steam Deck OLED made with a translucent shell for $679 — only in the U.S. and Canada.
Though let’s be real: This thing will sell like hotcakes and Valve will reconsider the definition of “limited supply.” Have I considered ordering one of these
limited editions? Yes. But I am also practicing restraint. Like, how many Steam Decks should one human have?

I suppose that’s the question, really. If you already have a Steam Deck, is the OLED model technically an incremental improvement? Yes. But you might find that
the screen and the battery justify the double dip.
</item 42>

<item 43>
The Verge’s 2023 holiday gift guide for kids (The Verge, 2023-12-18T23:14:07+00:00)
Price: $119.99 Amazon Best Buy

8BitDo SN30 Pro (Special Edition) The 8BitDo SN30 is more than a small controller best suited for tiny hands. It also oozes nostalgic vibes to make you feel a
bit of what mom and dad did while playing the original SNES — albeit, with modern games on PC or the Nintendo Switch. What’s more, its see-through color schemes
are sure to entice just about everyone, regardless of what year you were born.

Price: $44.99+ Amazon Walmart 8BitDo

Nintendo Switch Lite (Isabelle’s Aloha Edition) The Steam Deck may have ushered in a new era of handheld gaming, but it’s a bit pricey and overbearing for a
child. That said, there’s still no substitute for the simplicity of Nintendo’s Switch Lite and its travel-friendly 5.5-inch display.
</item 43>

<item 44>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
In early 2022, Valve released my favorite video game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of
my sprawling collection of video games wherever I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much
better games looked on our family’s OLED TV.

And yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.
</item 44>

<item 45>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Engadget

Hades is my most-played game on the original Steam Deck, so I used that to test out Valve’s battery life claims. Though it doesn’t support HDR, a game like
Hades pops on the OLED, and its animations look more fluid than on the LCD version. The OLED lasted four hours and 23 minutes before dying. With the same
settings (800p with a 60 fps cap and similar brightness) the original model died after two hours and 42 minutes. That’s a 62 percent improvement, surpassing
Valve’s own estimates, though batteries degrade and I’ve had my LCD Steam Deck since the device’s launch in early 2022. In a stress test that involved playing
Elden Ring with everything maxed out, the OLED battery lasted two hours and 20 minutes, about 40 percent longer than the original.

Playing the Steam Decks back-to-back highlighted another significant quality-of-life improvement: heat management.
</item 45>

<item 46>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The HDR OLED model is what they would have released the first time around, if the technology had existed before 2022. The handheld market has evolved quickly —
in just over a year, the Steam Deck proved there’s widespread demand for handheld PCs, and its success helped drive chip makers and display manufacturers to
build hardware specifically for high-fidelity mobile play. This is great news for Valve and the Steam Deck, and also the other handhelds coming out of companies
like Ayaneo, ASUS, Ayn, Logitech, Retroid and Razer. Hell, maybe even Playdate will get an HDR OLED display one day. (Just kidding; Playdate is literally
perfect).

So, no, the HDR OLED refresh isn't the Steam Deck 2, but it is a welcome improvement on an already good device. The Steam Deck OLED is the facetuned,
photoshopped, spit-shined version of Valve’s handheld, and it features tangible, clever improvements.
</item 46>

<item 47>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It delivers on the promise of building a hacker-samurai in a neon-infused open world replete with futuristic heists, daunting choices, and striking characters.
Phantom Liberty and Patch 2.0 may not completely erase the memory of CD Projekt’s initial botched release, but they come pretty damn close. Three years after we
first set foot in V’s shoes, Cyberpunk 2077 has finally justified the hype. —M. Mahardy

14. Dead Space

Developer: Motive Studio

Where to play: PlayStation 5, Windows PC, and Xbox Series X

With The Last of Us on HBO and Resident Evil 4 back in the conversation, it’s already a banner year for survival horror. Motive Studio’s Dead Space remake is no
exception. Following in the footsteps of Capcom’s aforementioned title, the original Dead Space brought the third-person-action focus of Resident Evil 4’s
formula to a deteriorating ship in outer space.
</item 47>

<item 48>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
Update (Nov. 17): Checked prices and links for accuracy.
</item 48>

<item 49>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
At the risk of diminishing its accomplishments, I’m going to make a comparison that should cut to the quick for interested parties: This is the most A Short
Hike-alike that I’ve discovered yet, and I’ve been chasing that singular high for years. So if you have a lazy weekend afternoon and want to spend it playing —
not gaming, but playing, in the joyous, unstructured sense of a day in the park — I can’t recommend this li’l game enough. —Chris Grant

Related Lil Gator Game has the heart of Wind Waker and the adventure of Breath of the Wild

34. Starfield

Developer: Bethesda Game Studios

Where to play: Windows PC and Xbox Series X

Starfield had so much to prove, it’s easy to lose sight of what it accomplished: top-tier world-building, a fantastically customizable ship builder, and
Bethesda’s most engaging combat to date.
</item 49>

<item 50>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—M. Mahardy

17. Dredge

Developer: Black Salt Games

Where to play: Nintendo Switch, PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

Dredge is a Lovecraftian horror experience masquerading as a simple fishing game.

The open ocean is filled with terrible creatures that can and will damage or destroy your boat. And each of the major islands you visit comes with its own evil
sea beasts that must be dealt with if you want to progress the story or fish peacefully. Dredge ultimately tells a dark parable about loss, and how obsession
can own you if you aren’t careful.

But what makes Dread so special — and one of the best games of 2023 — is that under its foreboding story and twisted environments is a fishing game that grows
more complex with every outing, centered around an upgrade system that feels amazing to progress through. As the dangers around you grow, so too does your
ship’s capabilities.
</item 50>
=== END ITEM id=b1023334eaed0eb0 ===

=== ITEM id=f4e60ed5973ec1d6 ===
Gold sentences and facts:
Sentence 0: The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.
  Fact 0.0: The agreement also meant that Apple never switched to a Google competitor.
  Fact 0.1: The agreement also meant that Apple never allowed users to choose their browser when setting up their iPhones.
Sentence 1: Apple has already indicated that it plans to appeal, but appealing is a lengthy process that can take around 18 months.
  Fact 1.0: Apple has already indicated that it plans to appeal.
  Fact 1.1: Appealing is a lengthy process.
  Fact 1.2: Appealing can take around 18 months.
Sentence 2: They use their store, their payments, they force developers to all have the same terms, they force OEMs and carriers to all have the same terms.
  Fact 2.0: The party described uses its own store.
  Fact 2.1: The party described uses its own payments.
  Fact 2.2: The party described forces developers to all have the same terms.
  Fact 2.3: The party described forces OEMs to all have the same terms.
  Fact 2.4: The party described forces carriers to all have the same terms.

Retrieved items:
<item 1>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 1>

<item 2>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Like Masimo, AliveCor argued that Apple stole its tech to develop the wearable EKG feature, thereby infringing on its patents. However, while the two cases look
similar, the result of one doesn’t speak to how the other will play out.

“Every case is different, every patent is different,” says Levi. “Different complainants who bring up a case have got their own businesses, their own products,
and those implicate a whole different set of facts.”

The Apple Watch SE will remain unaffected because it doesn’t have a blood oxygen sensor. Photo by Amelia Holowaty Krales / The Verge

Case in point, while the ITC granted AliveCor’s import ban, it hasn’t had any impact because the Patent Trial and Appeal Board ruled that AliveCor’s EKG tech
isn’t actually patentable. AliveCor would have to win its appeal to that ruling before any Apple Watches with EKG features would be pulled from the market for
that reason.
</item 2>

<item 3>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
And after the 24th, they’ll disappear from Apple Stores, too.

“I think Apple sees the writing on the wall and they’re preparing themselves,” says Ben Levi, a partner at Levi Snotherly & Schaumberg, which has experience
litigating ITC disputes.

It’s rare to see Big Tech lose — and with such tangible consequences at that. There are still ways that Apple can keep the watches on sale, but it’ll likely
take a lot of waiting or a lot of money to make it happen.

How Apple got here

This ban is the result of a long-standing patent dispute between Apple and the medical device maker Masimo. The latter is known for its pulse oximetry tech,
generally referred to as SpO2 or blood oxygen saturation in the wearable world, and it claims that Apple is using that patented technology without permission.

This particular story started about 10 years ago when Apple reached out to Masimo about a potential partnership around blood oxygen features on its wearables.
</item 3>

<item 4>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What would you say the differences are between the Apple and Google cases?

I would say Apple was ice and Google was fire.

The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the
same terms, they force OEMs and carriers to all have the same terms.

Whereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re
paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and
it’s right there for everybody to read and to see plainly.

I think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing,
whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute.
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
</item 6>

<item 7>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
“From a legal perspective, whatever change Apple makes has to be significant enough to avoid the patent, and it cannot be what’s called a ‘colorable’
difference. Minor changes will not be sufficient,” Iancu says.

Basically, Apple can’t simply make a small tweak to the code and call it a day. Masimo claims that the patents are hardware related, so even completely
disabling the sensor may not work depending on how the patent is written. Apple could remove it entirely from the design, but that would require mucking around
with Apple’s meticulous supply chain and could take some time.

Then there are the options that would cost Apple some money. Apple could sit down with Masimo and hammer out a licensing deal for the infringed patents. In a
New York Times interview, Masimo CEO Joe Kiani said Apple “had not engaged in licensing negotiations.” It’s an easy-seeming fix, so why not do it? For starters,
both parties have to want to settle.
</item 7>

<item 8>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
</item 8>

<item 9>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
</item 9>

<item 10>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 10>

<item 11>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
For starters, both parties have to want to settle.

“Apple is a tough nut to crack,” says Brittingham, noting that Apple is one of the most sued companies in Silicon Valley. The company’s large cash reserves also
make taking Apple to court a costly endeavor. Masimo, for example, has reportedly spent $60 million litigating the issue thus far. “I think they tend to hold
out as long as possible in order to make it less appetizing to sue Apple.”

The other option is for Apple to avoid the import ban altogether. Iancu says Apple could simply decide to manufacture the watches within the US instead of
overseas — but that’s a significant supply chain change that’s unlikely to happen any time soon.

Does this set a precedent?

This import ban isn’t the only one Apple is facing. In February, the Biden administration also declined to veto another ITC import ban and cease-and-desist
order stemming from a dispute with AliveCor over the Apple Watch’s EKG features.
</item 11>

<item 12>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“We try to get people up and running as fast as possible.”

Cue acknowledged that the ISA didn’t allow Apple to offer users a choice of search engines during setup but also said he wouldn’t want to do that anyway. “We
try to get people up and running as fast as possible,” he said. “Setup is just critical stuff.” Showing people a bunch of search engines they’ve never heard of
would just be a bad user experience, he argued; even Cue couldn’t remember the names of some of the alternatives to Google. “We make Google be the default
search engine,” he said, “because we’ve always thought it was the best. We pick the best one and let users easily change it.” (“Easily” is a persistent point of
contention in this trial — DuckDuckGo’s CEO, who testified last week, claimed it takes “too many steps” to switch.)

As for the privacy pop-ups? This is where Bellshaw began to press on how exactly Apple decided Google had the best product.
</item 12>

<item 13>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 13>

<item 14>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Where does Apple go from here?

A hail mary veto is unlikely, but that doesn’t mean Apple is just going to accept an import ban on a $17 billion segment of its business. Apple spokesperson
Nikki Rothberg told The Verge in a statement that the company was “pursuing a range of legal and technical options to ensure that Apple Watch is available to
customers.” That means the watch is going back on sale one way or another — it’s just a question of what path Apple takes.

Apple has already indicated that it plans to appeal, but appealing is a lengthy process that can take around 18 months. The company will also probably ask for a
stay on the import ban, but Brittingham says that’s not likely to be granted because Apple would have to prove that it would be irreparably harmed. In other
words, it’d have to make a convincing case that it would go out of business if it can’t sell the Apple Watch.

Apple says it’s exploring both legal and technical options to get around this import ban.
</item 14>

<item 15>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 15>

<item 16>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Apple getting a presidential veto would be like lightning striking the same place twice.

“It is extraordinarily rare for the President to overturn a decision from the International Trade Commission and in my view, [it’s] unlikely in this particular
case,” says Andrei Iancu, co-chair of the Council for Innovation Promotion (C4IP) and partner at Sullivan & Cromwell LLP, who was also the director of the US
Patent and Trademark Office from 2018 to 2021.

To get a veto, a company usually has to show a public interest or health policy basis — and that isn’t the case here. “It’s unlikely that the import exclusion
order will be disapproved by the USTR and the president,” says Smith Brittingham, partner at Finnegan, Henderson, Farabow, Garrett & Dunner, LLP. “The only real
answer would be, ‘Well, it’s Apple.’”

That said, Apple did manage to beat the system once. In 2013, then President Barack Obama vetoed an import ban on the iPhone.
</item 16>

<item 17>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 17>

<item 18>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Earlier this year, the US International Trade Commission (ITC) ruled that Apple had infringed on two patents from medical device maker Masimo. As a result, the
ITC said it would impose an import ban on the Apple Watch Series 9 and Ultra 2 starting December 26th. At the time it was hard to believe that this would
actually come to pass: Apple gets sued all the time, and even when it loses, how often does it actually face dramatic consequences?

Well, now would be one of those times. Experts say that, barring a Christmas miracle, it’s unlikely that Apple will find a way to escape the ban. Case in point,
the company shocked everyone yesterday when it decided to preemptively pull the watches from its online store starting December 21st at 3PM ET. And after the
24th, they’ll disappear from Apple Stores, too.
</item 18>

<item 19>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Thou shalt not have a competing store on iOS and thou shalt not use a competing payment method. And I think Apple should be receiving at least as harsh
antitrust scrutiny as Google.

It’s interesting to me that because Google distributes the Android operating system as open source, they had to put all these deals out in the open. More out in
the open, I should say — certainly they still wanted to keep them secret.

But I’m going down my story about all the best emails from the Epic v. Apple trial — and we do have a lot of documents from both Apple and Google that show they
were similarly self-serving in terms of deals.

I’d say this is the thing that’s disappointed me the most with Apple and Google: even at the peak of the antitrust trial against Microsoft, Microsoft was
awesome to developers. Microsoft has always been awesome to developers, always being respectful, giving developers a great deal and treating them as partners,
you know? And so even as Microsoft was crushing corporate competitors, the developer experience was excellent.
</item 19>

<item 20>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
In 2013, then President Barack Obama vetoed an import ban on the iPhone. At the time, Samsung accused Apple of infringing on its cellular data patents. But in
that case, according to Levi, Apple was able to make the argument that there were issues regarding fair, reasonable, and nondiscriminatory licensing (FRAND).
The patents involved were considered standard and essential, and the Obama administration was concerned the import ban would give Samsung “undue leverage.”
Plus, it only impacted the iPhone 4 and older models of the iPad. Newer iPhones didn’t use an infringing chipset. In the case of the Apple Watch, Apple doesn’t
have any of these arguments.

“In that case 10 years ago, Apple really rolled the dice and the pulled the rabbit out of the hat. It’s very unlikely for them to do that again,” says Levi.

Where does Apple go from here?
</item 20>
=== END ITEM id=f4e60ed5973ec1d6 ===

=== ITEM id=47133d9b8057691d ===
Gold sentences and facts:
Sentence 0: Argentina, Japan, Chile, Samoa and Fiji have all provided highlights — Los Pumas are also in the semi-finals — and England have beaten all of them.
  Fact 0.0: Argentina has provided highlights.
  Fact 0.1: Japan has provided highlights.
  Fact 0.2: Chile has provided highlights.
  Fact 0.3: Samoa has provided highlights.
  Fact 0.4: Fiji has provided highlights.
  Fact 0.5: Los Pumas are also in the semi-finals.
  Fact 0.6: England have beaten Argentina.
  Fact 0.7: England have beaten Japan.
  Fact 0.8: England have beaten Chile.
  Fact 0.9: England have beaten Samoa.
  Fact 0.10: England have beaten Fiji.
Sentence 1: S﻿AINT-ETIENNE: The Wallabies will have to endure a more tortuous Sunday night watch than even the worst rom com as their Rugby World Cup fate is decided by Fiji and Portugal in Toulouse.
  Fact 1.0: This report is datelined Saint-Etienne.
  Fact 1.1: The Wallabies will have to endure a more tortuous Sunday night watch than even the worst rom com.
  Fact 1.2: The Wallabies' Rugby World Cup fate is decided by Fiji and Portugal.
  Fact 1.3: The Fiji and Portugal match is played in Toulouse.
  Fact 1.4: The Wallabies' Rugby World Cup fate is decided in Toulouse.

Retrieved items:
<item 1>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England replacements: Theo Dan, Ellis Genge, Kyle Sinckler, Ollie Chessum, Billy Vunipola, Danny Care, George Ford, Ollie Lawrence.

The Springboks are unchanged from that epic quarter-final with France, having all played their part as Jacques Nienaber's side hung on in the second half. The
contribution of backs Kurt-Lee Arendse, Damian de Allende and Cheslin Kolbe saw all three cross for first-half tries, with Kolbe's coming off the back of a
charged-down conversion from France's Thomas Ramos. Preventing those two points proved crucial in the end.

The key contribution, though, came from Eben Etzebeth, who bulldozed his way over to score the only five-pointer of the second half and wrestle the game back
into South African hands.
</item 1>

<item 2>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England vs. South Africa betting odds, prediction

The last time these two sides met was a tumultuous period to say the least, particularly for English rugby. South Africa's dominant 27-13 victory at Twickenham
proved to be the last game in charge for Eddie Jones, the coach who had taken England to the 2019 World Cup final. In his place, Steve Borthwick was hired, and
he's just one game away from repeating that feat.

England's form at this tournament has been perfect — at face value. Argentina, Japan, Chile, Samoa and Fiji have all provided highlights — Los Pumas are also in
the semi-finals — and England have beaten all of them. However, only Japan and Chile were defeated with bonus points, and the narrow wins over both Samoa and
Fiji in their last two games showed that England are certainly gettable.
</item 2>

<item 3>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England and South Africa face off in the 2023 Rugby World Cup's second semi-final, played at the Stade de France in Saint-Denis, France. The game kicks off at 9
p.m. local time on Saturday, October 21, 2023.

Here are the kick-off times for this game around the world:

Location Date Time USA Sat, Oct. 21 3:00 p.m. ET Canada Sat, Oct. 21 3:00 p.m. ET Australia Sun, Oct. 22 6:00 a.m. AEDT UK Sat, Oct. 21 8:00 p.m. BST New
Zealand Sun, Oct. 22 8:00 a.m. NZDT India Sun, Oct. 22 12:30 a.m. IST Singapore Sun, Oct. 22 3:00 a.m. SGT Malaysia Sun, Oct.
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The latter is winning only his seventh England cap but gets the starting nod alongside Maro Itoje, with Genge, Chessum and the rest of the replacements hoping
to make an impact sizeable enough to overcome the Springboks' infamous 'bomb squad'.

Their time is now 🙌



Here is the @EnglandRugby team to take on reigning champions South Africa in the semi-final in Paris#RWC2023 | #ENGvRSA pic.twitter.com/nkf6rXBPUU — Rugby World
Cup (@rugbyworldcup) October 19, 2023

England starting XV: Freddie Steward, Jonny May, Joe Marchant, Manu Tuilagi, Elliot Daly, Owen Farrell, Alex Mitchell; Joe Marler, Jamie George, Dan Cole, Maro
Itoje, George Martin, Courtney Lawes, Tom Curry, Ben Earl.
</item 6>

<item 7>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
They sit tenth in the table having won four and lost as many from their opening eight games.

But their most recent Premier League victory over Brentford may give the Red Devils hope their season is on the up.

Sheffield United vs Manchester United is live on talkSPORT - kick off is at 8:00pm

9 United got a dramatic win over Brentford before the international break Credit: Getty

An England vs South Africa double header

If you've had your football fix for the day, there's also a packed Saturday of rugby and cricket.

It's crunch time for England in the World Cups as they face South Africa in the semi-final of the Rugby World Cup and in the ODI Cricket World Cup.

After England secured victory over Fiji in the quarter-finals, they now look to overcome defending champions South Africa to advance into their second
consecutive Rugby World Cup final.
</item 7>

<item 8>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
So the team understands that and we have our belief in anything's possible at the World Cup.

READ MORE: ﻿'Blame myself': Eddie reveals great World Cup regret

READ MORE: Referee ripped after Junior Paulo's dominant KO

LIVE COVERAGE: All the latest updates from Bathurst 1000

Mark Nawaqanitawase runs the ball during a Wallabies training session at Stade Roger Baudras. (Getty)

"But also, we understand how strong Fiji is. And we're proud for them as a proud nation. For myself, I'll be proud of it either way it goes. We'll be ready if
it comes our way."

Kerevi was photographed holding his face in disbelief in one of the defining images of the World Cup ﻿after the Wallabies were pumped 40-6 by Wales in Lyon.
</item 8>

<item 9>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
Sarina Wiegman talks a lot about defending against counterattacks, because this is basically the key to the whole enterprise. Cut off the counter, and with
England’s technical quality they can easily dominate. It is also the essential measure of how the team is functioning as a collective: the efficiency of the
press, the closing of the angles, the command of space. Lineth Beerensteyn’s two goals were the sort England simply were not conceding four months ago.

Mary Earps shows her frustration after conceding a second goal to the Netherlands. Photograph: Tom Jenkins/The Guardian

So England passed the ball around a bit and gave it away a bit and essentially ran around trying to feel normal again, and on countless occasions Kelly or Hemp
would win a first or second ball, look up and see a penalty area empty of white shirts.
</item 9>

<item 10>
Thirty-seven phases of agony define Ireland’s greatest heartbreak as curse continues (The Independent - Sports, 2023-10-14T21:36:10+00:00)
And then there were those 37 phases… the most painful finish in Irish rugby history.

Ireland were left bereft after an agonising defeat (PA Wire)

Instead of being feted as the heroes to finally overcome that insurmountable obstacle, the Ireland players will still have to answer endless questions about the
heavy burden of previous knockout failures and fans will endure at least four more years of choking jokes at their expense.

As for the All Blacks, they looked lost at times during this World Cup cycle and it’s still faintly remarkable that head coach Ian Foster survived last summer
as the home defeats to Ireland, South Africa and Argentina piled up.

Yet somehow, he hung on and turned things round. He has the unwanted title of being the first New Zealand coach to lose a World Cup pool match but now a
semi-final awaits next Friday evening against an Argentina side that, frankly, are a tier below them. Forget the hand-wringing ahead of the tournament, the All
Blacks are one very small step away from another World Cup final.
</item 10>

<item 11>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
TV channel Streaming USA — Peacock Canada TSN TSN+ UK ITV ITV X Australia Channel 9 Stan Sport, 9Now New Zealand Sky GO Sky GO India — Fan Code Hong Kong BeIN
Sports BeIN Sports Malaysia BeIN Sports BeIN Sports Singapore BeIN Sports BeIN Sports

MORE: Every single team to have lifted the Rugby World Cup trophy

England vs South Africa lineups, team news

"The start is crucial, the end is vital..."



- Steve Borthwick explains changing 3️⃣ of his England starters for Saturday's Rugby World Cup semi-final, from Liam Heagney 👨‍💻 in Paris. #RWC2023 #ENGvRSA
#EnglandRugby
</item 11>

<item 12>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
Oct 8th, 2023

Oct 8th, 2023

S﻿AINT-ETIENNE: The Wallabies will have to endure a more tortuous Sunday night watch than even the worst rom com as their Rugby World Cup fate is decided by
Fiji and Portugal in Toulouse.

Australia will gather together in their plush Saint-Galmier hotel at 9pm local time (6am Monday AEDT) to view what will almost certainly be their World Cup
execution.

The Wallabies are currently sitting second in Pool C but a single bonus point would be enough for Fiji to leapfrog them and advance to the quarter-finals in
Marseille.

Watch all the action from Rugby World Cup 2023 on the home of rugby, Stan Sport.
</item 12>

<item 13>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
NEW PODCAST! Michael Atkinson and Sam Worthington preview a thrilling final round of pool games to decide the quarter-finalists at the Rugby World Cup in
France﻿

He wanted to regain form at club level ﻿with Urayasu D-Rocks but his hand remained up for Test selection.

"I chose to leave Australian shores and understand the ruling around that and what it holds," he said.

"I've got to perform for my club first and if you get the call up, you get the call up. If you don't, I'll take my three months break and enjoy that."

Kerevi is an unabashed Jones fan.

Eddie Jones with fans during a Wallabies training session at Stade Roger Baudras. (Getty)

"He's up at 4am working hard. We catch him in the gym, he's working hard and I swear he doesn't even sleep.
</item 13>

<item 14>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The last time Cheslin Kolbe played England at a Rugby World Cup... 🇿🇦 🏆#RWC2023 | #ENGvRSA pic.twitter.com/Riw2uaSaKa — Rugby World Cup (@rugbyworldcup) October
19, 2023

Both sides' journeys have seen twists, turns, and bumps in the road, but it's led them both to within 80 minutes of a Rugby World Cup final. England will no
doubt be looking to echo the spirit of 2019, when they beat the All Blacks 19-7 to reach the final, but in South Africa, they face a side who have taken their
game to new heights just to reach the semis.

MORE: 2023 Rugby World Cup final: Details, kick-off time and every team's route

What time does England vs South Africa kick off?
</item 14>

<item 15>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The Springboks have done it tough, too, having already faced two of the tournament favourites on their way to the semi-finals. Comfortable defeats of Scotland,
Romania and Tonga couldn't mask the disappointment of the 13-8 loss to Ireland, which consigned South Africa to second in Pool B. However, with Ireland sent
home by the All Blacks in the quarters, the Bokke showed their World Cup mettle to edge the hosts 29-28 at the Stade de France, sneaking past Les Bleus in one
of the best games of rugby you're ever likely to see.
</item 15>

<item 16>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
Speaking after the match, Buttler admitted his team had been ‘short of our best by a very long way’, saying there is ‘no clear answer’ as to how to turn their
fortunes around.

“It’s incredibly tough, an incredibly disappointing tournament. As captain, you feel that a lot,” he said.

Advertisement

“We’ve been short of our best by a very long way. I’m disappointed for myself and the boys that we’ve not given a good account of ourselves… if there was one
golden nugget that we’re not doing then we’d pick that up.”

Sports opinion delivered daily

England’s tournament may only stay mathematically alive until Sunday evening (AEST), when they face the daunting challenge of tournament hosts India in Lucknow,
with defeat certain to end their semi-final chances for good.
</item 16>

<item 17>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The 2019 Rugby World Cup final between England and South Africa was one of many firsts: the game itself capped off a sensational tournament played outside of
the sport's traditional strongholds for the first time; it was the first final in which South Africa scored a try, and by full-time, the 32-12 score in the
Springboks' favour sealed a record-equalling third title for the Rainbow Nation.

Fast forward four years, and South Africa once again prepare to take on England in an all-or-nothing match to decide which nation gets a shot at World Cup glory
— but this time around, the pair meet in the semi-finals of the 2023 tournament, rather than the decider.

That won't make an ounce of difference for either camp, though, as these two long-time rivals prepare for a battle that could re-define the definition of a
'physical encounter.'
</item 17>

<item 18>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
In short, this semi-final requires a huge step up in quality for Borthwick's side, and that's why we think that the Springboks will be victorious at the Stade
de France. South Africa have already shown they can match, and beat, the world's best, and we think that they'll do the latter in Saint-Denis.
</item 18>

<item 19>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
He and Franco Mostert pack down together in the second row, while hooker Bongi Mbonambi, utility Deon Fourie, veteran Duane Vermeulen and the devilish Kwagga
Smith will all be looking to back up superb quarter-final performances to take the Springboks to a second consecutive World Cup final.

South Africa starting XV: Damian Willemse, Kurt-Lee Arendse, Jesse Kriel, Damian de Allende, Cheslin Kolbe, Manie Libbok, Cobus Reinach; Steven Kitshoff, Bongi
Mbonambi, Frans Malherbe, Eben Etzebeth, Franco Mostert, Siya Kolisi, Pieter-Steph du Toit, Duane Vermeulen.

South Africa replacements: Deon Fourie, Ox Nche, Vincent Koch, RG Snyman, Kwagga Smith, Faf de Klerk, Handre Pollard, Willie le Roux.
</item 19>

<item 20>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
Every match ad free, live and on demand in 4K UHD

The most likely outcome is that Fiji wins easily and the Wallabies will be pushing confirm on their off-season holiday plans.﻿

"We've only got ourselves to blame, we put ourselves in this position," Wallabies centre Samu Kerevi, who was born in Fiji, said after an open training session
in front of 1000 keen locals at the Andrezieux-Boutheon rugby club on Saturday.

"We'll sit down as a team and watch together on Sunday night. I think we understand the situation ... our results and the way we played, we put ourselves in
this position. So the team understands that and we have our belief in anything's possible at the World Cup.
</item 20>

<item 21>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
talkSPORT will have updates from all the action in Paris as England take on the Springboks from 8:00pm

9 Owen Farrell will be looking to inspire England to victory against the Springboks Credit: Getty

Over in India, England's Cricket World Cup hopes hang by a thread after their latest shock defeat to Afghanistan.

They will be boosted by the news that Ben Stokes is fit for the match after he picked up a hip injury five days before the tournament.

But victory for England could scarcely be more crucial.

Having also lost to New Zealand in their opener, their chances of finishing in the top four and progressing to the semi-finals are looking dubious.

But South Africa, despite winning their first two games, suffered a shock defeat to the Netherlands in their most recent match - a sign for England that the
Proteas can be beaten.
</item 21>

<item 22>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
after newsletter promotion

And clarity is perhaps what this team has been lacking most of late: clarity of selection, clarity of formation, clarity of mission and motivation. How much
prestige should we really attach to the Nations League, anyway? Is it primarily an Olympic qualification shootout or something worth winning in its own right?
England ended up trying to answer these questions on the hoof, and somehow it was only when their options had finally clarified down to one – survival or
oblivion – that they recovered their poise and verve.

This entire post-pandemic international cycle – a Euros, a World Cup, an Olympics and another Euros squeezed into three years – feels like a kind of sadism
dressed up as a panorama of opportunity. There is a school of thought out there that a fallow summer might not have been the worst thing for this team: time to
reflect and process, time to breathe, time to refresh.

Instead, they now travel to Glasgow on Tuesday for one last push. The minds are still frazzled.
</item 22>

<item 23>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
"Last week 45,000 fans (versus Portugal), last night (60,000 in Lyon). Rugby's such a small aspect in terms of Australians but our fans have turned up here,
we've seen them and spoken to them and their unconditional love for us, the fact that they came 20,000km to get here and support us, has been awesome.

"My message is just to understand that we're working extremely hard, there's no stone unturned, especially the youth in this team, to leave the jersey in a
better place and just to stick with us.

"I know the results haven't been coming our way the last couple of years but when it does turn, that's what we're building towards.﻿"
</item 23>

<item 24>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
It’s such a lame excuse.”

Atherton described the England players as looking ‘dazed and shell-shocked, unable to comprehend what has befallen them’, and wrote in The Times that the Sri
Lankan defeat officially represented the ‘end of an era’ for English white-ball supremacy.

Advertisement

“Each defeat, now four in five matches, has chipped away at the confidence of a group of players at the end of a cycle,” Atherton wrote.

“No batsman looks in any kind of form. Every player has been used; some discarded and used again. They lost faith in their all-rounder strategy before the South
Africa match, making three changes, and then went back to it here, making three more.

“Players sense uncertainty in decision-making – and there has been a lot of uncertainty.

“It was hard to think that they could put in a worse performance than in Mumbai but they managed it with the bat in Bengaluru.”
</item 24>

<item 25>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
“That judgment will be decided by the 2027 World Cup.”

He said the team – including coaches and players – have already completed their own review of the World Cup disaster. The next one would be “a formal review to
be done by independent consultants” and he was ready to participate.

“I didn’t come back to Australia to have a holiday – sit down at Coogee Beach, eat fish and chips, have a nice flat white. It was always going to be a battle.
When you’ve got a team that hasn’t done well for a long period of time, it’s always a battle.”

Advertisement

Jones said it was crucial that Rugby Australia got alignment in the code from grassroots through to the top team as it looks to rebuild it’s international
standing. Australia, a two-time world champion, is currently ranked 10th in the world.

“You just got to watch the quarterfinals at the weekend. We’re not at that level.
</item 25>

<item 26>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
Click below for more ⤵️ https://t.co/LeUuw4kPfP — RugbyPass (@RugbyPass) October 19, 2023

Marcus Smith hardly put a foot wrong at fullback against Fiji, but the world champions are an entirely different proposition for England to deal with at the
back. The high-ball security and hulking frame of Freddie Steward returns to the No. 15 jersey in one of three changes, while at inside centre, Manu Tuilagi
starts hoping to repeat his heroics of four years ago, when his early try in a dominant semi-final performance took England past the All Blacks.

England's other two changes come in the tight five, with Steve Borthwick making some big calls ahead of this game. Joe Marler starts ahead of vice-captain Ellis
Genge at loosehead prop, and Ollie Chessum is replaced in the second-row by George Martin.
</item 26>

<item 27>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
But I don't think people really understand demons, everyone's fights in between.

"Eddie's a human as well and at the end of the day it's a results-driven sport, you know, we need results to get that. But he's committed to us and we love that
about him.

"He hasn't thought about going home, he's thought about how we can contribute best and that's what we've done these last three days. I feel for him, you know,
like as a head coach, you maybe can't show that side of himself.

Eddie Jones and Will Skelton talk at Stade Roger Baudras. (Getty)

"But he's been so strong for us and gives us strength to just be positive. And again, we put ourselves in this situation."

Wide World of Sports also asked Kerevi to deliver a message to disillusioned Wallabies fans.﻿

"I think at home there's so many sports going on and some saying rugby's dying, but rugby's massive here overseas," he said.
</item 27>

<item 28>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
We’re not at that level. And we can’t pretend to be at that level. But can we be at that level by 2027? Yes, we can,” said Jones.

“We’re definitely off the pace but the opportunity to get better is there. I love the game. I love working with young players. I loved seeing some of the young
players maturing during the World Cup and some of them struggle. And that’s all part of it. But I love the game and I love coaching the Wallabies.”

Jones was named Wallabies coach at the start of the year as replacement for Dave Rennie. He came in with grand statements that his team could win the Rugby
Championship, Bledisloe Cup and make a smash and grab on the World Cup.
</item 28>

<item 29>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
22 3:00 a.m. SGT Malaysia Sun, Oct. 22 3:00 a.m. MYT Hong Kong Sun, Oct. 22 3:00 a.m. HKT

England vs South Africa live stream, TV channel

TV networks worldwide will be showing all 48 matches at the tournament live, including many on free-to-air television.

On social platforms alone, video content from the Rugby World Cup and organisers World Rugby drew more than 2.1 billion views in 2019 — almost six times the
previous figure.
</item 29>

<item 30>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The #Springboks are gearing up for another tough challenge this week with a semi-final clash against England 🇿🇦🏴󠁧󠁢󠁥󠁮󠁧󠁿#StrongerTogether #RWC2023
pic.twitter.com/HLSnnjWggB — Springboks (@Springboks) October 18, 2023

England are the only side still competing who are yet to lose a game, and that will stand them in good stead coming into this semi-final. Dominant wins over
Argentina, Japan and Chile sealed passage to the last eight, and after being given an almighty scare by Samoa in their final pool match, England overcame a
Fijian fightback in Marseille to triumph 30-24. Despite being outscored by three tries to two, captain Owen Farrell kicked 20 points, sending his side into the
last four at the expense of 2023's breakout stars.
</item 30>
=== END ITEM id=47133d9b8057691d ===

=== ITEM id=d4802a8873e41fbe ===
Gold sentences and facts:
Sentence 0: These delays may lead to bets being refunded depending on the circumstances and the specific rules of your sportsbook.
  Fact 0.0: Delays may lead to bets being refunded.
  Fact 0.1: Whether bets are refunded depends on the circumstances and the specific rules of the sportsbook.
Sentence 1: Every sportsbook jumps on board when a line moves or odds change for a legitimate reason, like an injury or roster move.
  Fact 1.0: Every sportsbook adjusts when a betting line moves for a legitimate reason.
  Fact 1.1: Every sportsbook adjusts when odds change for a legitimate reason.
  Fact 1.2: An injury or a roster move is an example of a legitimate reason for a line or odds to change.
Sentence 2: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 2.0: Sportsbooks are also gathering this information.
  Fact 2.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.

Retrieved items:
<item 1>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
This includes your initial $10 wager, along with a profit of $15.30.

It's essential to note that this calculation gives you the total payout, not just your earnings. If you're keen on tracking your profits from betting, remember
to subtract your original stake from the payout to get an accurate figure.

Why do moneyline odds change?

Lines at the best sports betting apps change quickly due to two primary reasons:

Public betting

New information

Before setting odds for an event, oddsmakers gather extensive data. Once they finalize the odds, they release them to the public.

As a bettor, you can bet immediately or wait for potential shifts in the odds, although predicting these shifts is challenging.

For instance, when examining odds for the next Super Bowl champion released shortly after the previous Super Bowl, these odds are based mostly on the recently
concluded season.

As the new season approaches and teams make changes, odds adjust based on perceived championship chances. As games progress, lines continue to evolve based on
team performances.

Public betting can also sway lines.
</item 1>

<item 2>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
Does overtime count in my moneyline bet?

Yes, in most sports and with most sportsbooks (including new betting sites), overtime (or any extra time or tiebreakers) does count in a moneyline bet.

That said, always check the specific rules of the sportsbook you're using to ensure you understand their policies.

Why do moneyline odds change?

Moneyline odds change for several reasons.

Public Betting Patterns: If a significant amount of money is being placed on one team or participant, sportsbooks might adjust the odds to balance their
liability. By making the odds less favorable for a heavily-bet team and more favorable for the less-bet team, they can encourage more even betting.

If a significant amount of money is being placed on one team or participant, sportsbooks might adjust the odds to balance their liability. By making the odds
less favorable for a heavily-bet team and more favorable for the less-bet team, they can encourage more even betting.
</item 2>

<item 3>
MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 (Sporting News, 2023-10-24T22:32:27+00:00)
By doing this research, you’ll find the best odds for the bets you were already planning on making, which means you can maximize your potential return on your
wager.

This can be a time-consuming process, but it also helps you boost your bankroll and give yourself the largest possible profit.

MLB betting rules at online sportsbooks

Every sportsbook has terms and conditions for all bets, MLB included. While those rules may be specific to each sportsbook, there are some general rules that
should apply everywhere.

For example, MLB First 5 Innings bets are usually considered actionable after five innings of play, or 4.5 innings in the case of the home team having the lead.

Another rule many sportsbooks follow for baseball bets is weather-related, such as if a game is suspended due to rain or other weather conditions. These delays
may lead to bets being refunded depending on the circumstances and the specific rules of your sportsbook.

Here are some other rules found at online sportsbooks for MLB betting.

Wagers include extra innings unless specified otherwise.
</item 3>

<item 4>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Ensure the site has a responsible gaming section.

The overall user experience should be smooth and hassle-free.

Safety and Security:

Top NBA betting sites prioritize users' safety.

Ensure your personal information is secure.

Some sportsbooks excel in ensuring cybersecurity; always go for those with solid reputations.

How to find the best NBA lines

Not all sportsbook betting odds are created equal. If you're willing to put in some legwork, you can shop for the best NBA lines to maximize your potential
return.

Movement in the odds depends on multiple factors, including team news, injuries, public betting, recent play, and even the number of games a team plays in a
row.

Oddsmakers use all the available information to craft their lines, and you can find value in shopping around for the most favorable lines and odds.

You can use our NBA odds page (see link above) to find the best lines for your prospective bet, or you can open accounts at multiple sportsbooks and do the
shopping yourself.
</item 4>

<item 5>
Totals Betting - What is an Over/Under Bet? (Sporting News, 2023-11-01T22:08:31+00:00)
Bookmakers adjust the line to balance the action and protect themselves from potential losses.

One of the most common reasons for line movement is a significant amount of money being wagered on one side (either over or under). Bookmakers adjust the line
to balance the action and protect themselves from potential losses. Injury Updates: If a key player is injured or unable to play, this can influence the
expected total score, leading bookmakers to adjust the line accordingly.

If a key player is injured or unable to play, this can influence the expected total score, leading bookmakers to adjust the line accordingly. Weather
Conditions: In outdoor sports, weather can play a significant role. For example, heavy rain or snow in a football game might lead to expectations of a
lower-scoring game, causing the line to move down.

In outdoor sports, weather can play a significant role. For example, heavy rain or snow in a football game might lead to expectations of a lower-scoring game,
causing the line to move down.
</item 5>

<item 6>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
However, you’d have a winner at the fourth. If you didn’t shop around, you’d never have found that half-point victory and be stuck with a push instead of a
profit.

What is the Vegas line?

Las Vegas sportsbooks set the standard for lines and odds across the country. If a sportsbook is copying a line, it’s usually copying the Vegas line. Most
online betting sites don’t stray too far from the Vegas line, using it as a foundation for setting all lines in every major sport.

If those lines do shift, it’s usually because of increased betting on one side of a bet. Shop lines locally, and you can take advantage when sportsbooks stray
from the Vegas line.

Why do the lines move?

It’s worth reiterating that sportsbooks want an equal amount of money bet on each side of every bet. When that happens, they earn the small profit built into
the odds without concern about who wins or loses.
</item 6>

<item 7>
MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 (Sporting News, 2023-10-24T22:32:27+00:00)
You’ll find live betting with top betting sites such as DraftKings, FanDuel, BetMGM, and Caesars.

The key to live betting is using a betting app to make your wagers. Apps allow you to track all of the changes — both on the diamond and in the odds — as they
happen. You won’t find that response time at a brick-and-mortar sportsbook.

Some common MLB in-game betting lines includes:

Next team to score

Run line and moneyline odds

Total runs scored

Winner of the next inning

Player props, such as total hits or strikeouts

Bet on MLB run lines, moneylines, and totals

Three of the most popular bet types for Major League Baseball games are run lines, moneylines, and totals. In case you’re not familiar, let’s explain all three.

A moneyline wager is as straightforward as they come. You’re betting on which team you believe is going to win the game.
</item 7>

<item 8>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
New Information: Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an
event. When such information becomes public, sportsbooks may adjust the odds accordingly.

Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an event. When such
information becomes public, sportsbooks may adjust the odds accordingly. Professional Bettors: Large wagers from sharp bettors or professional gamblers can
cause the lines to shift. These individuals often have insights or information that can influence the odds.

Large wagers from sharp bettors or professional gamblers can cause the lines to shift. These individuals often have insights or information that can influence
the odds. Initial Miscalculation: Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on
one side might indicate that the initial odds were potentially "off," prompting an adjustment.
</item 8>

<item 9>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Line shopping is the key that unlocks the best odds and lines at top sports betting sites.

The process is as simple as checking the lines and odds for your prospective bet at multiple online sports betting sites. The result is quite often better odds
and improved lines.

What is line shopping, and why is it worth your time? Find the answers below in our complete guide to line shopping.

How do sportsbooks set odds and lines?

The best sportsbooks employ oddsmakers and oddsmaking services to set their lines and odds.

Other sports betting apps simply copy those lines. The oddsmakers and oddsmaking services use computer models and algorithms that consider analytics,
statistics, injury and roster news, performance, matchup data, and more to develop the lines and odds.

Betting lines and odds are then subject to change based on the betting public’s reaction and any additional information that comes to light ahead of the game.
Sportsbooks will move the line or change the odds if the public bets heavily on one side.
</item 9>

<item 10>
MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 (Sporting News, 2023-10-24T22:32:27+00:00)
Traditional bet types, such as moneylines, where one bets on the outright winner; run lines, which focus on the margin of victory; and totals that predict
combined team scores, are all readily available.

For those who relish a deeper drop into the game, BetMGM has a deep lineup of player and team prop bets. Whether it's forecasting a player's strikeouts in a
game or predicting the cumulative runs scored by a team, there's something for every strategy.

For the long-term visionaries, BetMGM rolls out an impressive array of futures bets. Fans can place wagers on outcomes such as which team will clinch its
division or which player will be honored as the league's Most Valuable Player.

See more:

Caesars Sportsbook

Caesars has a comprehensive suite of MLB odds to elevate the game-watching experience. That includes run lines, predicting the margin of victory; totals, where
bettors forecast the combined scores; and player props, focusing on individual milestones such as hits, home runs, or RBIs.
</item 10>

<item 11>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks will move the line or change the odds if the public bets heavily on one side.

Every sportsbook’s goal is to even out the betting on both sides of every bet. That allows sportsbooks to earn the profit built into the odds and lines without
worrying about which side wins.

The betting is rarely the same at every sportsbook. While one sportsbook may be forced to move a line or change the odds because of betting, another won’t.
That’s when you’ll spot a difference between sportsbooks. This difference may allow you to book a bet at improved odds or a more favorable line.

The lines and odds may even be different right from the jump. Either way, the differences are why line shopping always makes sense. Shop around, and you can
find opportunities to lay less, win more, and improve your chances.
</item 11>

<item 12>
MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 (Sporting News, 2023-10-24T22:32:27+00:00)
Betting apps offer real-time updates on all odds, which allows you to react to game flow as it’s happening.

Your MLB betting app can help you track shifting odds and facilitate live betting. Betting apps offer real-time updates on all odds, which allows you to react
to game flow as it’s happening. Bet from within legal states: As long as you are in a state that has legal online sports betting, you can use your app to make
bets, track your account, add or withdraw funds, and more. Remember that you can only place bets when you are in the state lines, but you can access your
account and monitor wagers from anywhere you have internet access.

MLB live betting

The best sports betting apps offer live, in-game betting for Major League Baseball.

Live betting means you can bet on a game in progress, right down to what the next batter will do with the next pitch or what team will win the inning.
</item 12>

<item 13>
MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 (Sporting News, 2023-10-24T22:32:27+00:00)
This bet can be a fun way to keep tabs on certain players throughout the entire season. Home run leader: Which player do you think will crank out the most
homers during the regular season?

Which player do you think will crank out the most homers during the regular season? Regular-season win totals: Make an over/under wager on a team based on the
number of wins a sportsbook sets for them. Do you think your team will finish over that predicted win total, or will they come in under it?

Every sportsbook offers its own selection of futures and props bets, so shop around to find the sportsbooks that best fit your needs.

How to find the best MLB lines

It stands to reason that bettors always seek the most favorable MLB odds. “Line shopping” involves researching several different sportsbooks to compare odds.

This might involve using an odds feed, like the ones you’ll find here, or by comparing the odds on multiple online sportsbooks or MLB betting apps.
</item 13>

<item 14>
MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 (Sporting News, 2023-10-24T22:32:27+00:00)
You’re betting on which team you believe is going to win the game. Each team is given odds by the sportsbook to indicate their perceived chances of winning. The
underdog is represented by a positive (+) number, and the favorite is represented by a negative number (-).

Similar to a point spread in other sports, the run line bet gives the underdog a handicap of 1.5 runs in the game. The underdog needs to lose by one run or win
outright for bets on them to be successful, while the favorite in the game needs to win by two runs or more to cover the spread.

Totals bets, also known as over/unders, focus on the combined number of runs both teams score. Oddsmakers assign a total number of runs, and bettors must decide
if the two teams will score over or under that predicted total. The actual winner of the game doesn’t matter.

You’ll find these staple bet types at every online sportsbook that supports MLB betting.
</item 14>

<item 15>
MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 (Sporting News, 2023-10-24T22:32:27+00:00)
Wagers include extra innings unless specified otherwise.

The game must start on the scheduled day (local stadium time) for bets to have action.

The game must go 9 innings or 8½ if the home team is winning for the run line to have action.

On ‘team to score first’ wagers, once a run is scored, the prop will be graded, regardless of how long the game lasts.

Wagers on postseason baseball series are based on the team to win the series, regardless of the number of games involved.

For player props, the player must be in the starting lineup for wagers to have action.

It’s important to read through the rules for baseball betting before placing your MLB bets, and you can find them in the terms and conditions section of your
sportsbook’s website.

Related Pages:
</item 15>

<item 16>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Do point spread odds change?

Yes, point spread odds can change, and these shifts are commonly referred to as "line movement."

Lines move when the point spread or the odds associated with a particular bet change after sportsbooks have initially set them. There are several reasons for
these adjustments.

One of the most common causes is an imbalance in the amount of money being wagered on one side of the bet. If a significant majority of bettors are placing
their money on one team to cover the spread, the top sports betting apps will adjust the point spread to make the other side more appealing, thereby balancing
the betting action.

Another factor that can affect the spread odds is the occurrence of unexpected events, such as injuries to key players, changes in team strategies, or even
significant weather events that might impact a game's outcome.

For instance, if a star quarterback is announced as injured just days before a big game, the point spread might shift in favor of the opposing team.

Moreover, information and public perception play a role.
</item 16>

<item 17>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on one side might indicate that the
initial odds were potentially "off," prompting an adjustment. Event Dynamics: For live betting, where bets are placed while the event is in progress, the
evolving dynamics of the match can lead to frequent changes in the odds.

Sportsbooks adjust moneyline odds to ensure they remain profitable, balance their book, and reflect the most current information about the event.

Related Pages:
</item 17>

<item 18>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Moreover, information and public perception play a role. If a piece of news or a prevailing sentiment starts to gain traction about a particular team's
increased chances of winning, this can influence how people bet, leading sportsbooks to adjust their lines accordingly.

Point spread odds can and do change based on a combination of actual betting patterns, unexpected events, and shifts in public perception. Savvy bettors keep a
close eye on these line moves, as they can offer insights into where the value lies in a particular bet.
</item 18>

<item 19>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
What makes NBA betting special or unique?

Basketball odds are known to fluctuate more than those of other sports. The factors that play into those lines shifting can range from having only a few impact
players per team to the sheer frequency of games.

By following a single team, you can track its momentum swings, which players tend to get on hot or cold streaks, or how well they play at home vs. on the road.
You can use that information to your advantage for both pre-game and live wagers.

Similar categories of moneylines, spreads, totals, and NBA prop bets should appear across both formats.

The NBA is also special because of the number of bets available for every game, ranging from pre-game to live betting and parlays.

When you add up the standard bets, game props, player props, in-game betting, and more, there’s a staggering volume of bets for every NBA game at online
sportsbooks.
</item 19>

<item 20>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks want even betting so much that they’ll move a line or change the odds to draw more bets to the other side to get it that way.

That’s when you’ll see a difference between the lines and odds. Every sportsbook jumps on board when a line moves or odds change for a legitimate reason, like
an injury or roster move. It’ll be difficult for you to find value in that case.

But when a line moves or odds change because of betting patterns, only the sportsbook that saw the betting makes the change. That’s your opportunity right
there.

Line shopping is a tool to help you spot changes due to betting patterns. Then, you can simply take advantage of movement in a line or changes in the odds.

Choose a side, shop around at online sportsbooks, compare the lines and odds, and then pounce when you find one that has something different and more favorable
to your viewpoint.

See more: What is a parlay bet?
</item 20>
=== END ITEM id=d4802a8873e41fbe ===
