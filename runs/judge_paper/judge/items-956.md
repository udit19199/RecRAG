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

=== ITEM id=0364e0516f8f9761 ===
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
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
All of the buttons, triggers, and sticks feel just as good to use as any modern controller from a console manufacturer.

Like several other 8BitDo controllers, this one can be thoroughly customized with its Ultimate software on mobile devices, or on PC. From there, you can remap
controls, assign buttons to its two rear paddle buttons, tweak the actuation sensitivity of its triggers, and adjust stick settings. The controller allows up to
three custom control schemes, which you can switch between with its middle “star” button.

The Ultimate controller ships with a charging dock, and nesting inside of that dock is a 2.4 GHz transmitter that makes it fast and easy to connect to a PC. The
dock makes for a convenient way to charge the controller, and you can switch between DirectInput and XInput by using a switch on its back, depending on the
requirements of each PC game.

8BitDo makes a pricier version of this controller with Nintendo-style face buttons called the Ultimate Bluetooth. They’re virtually identical, and it can work
via 2.4 GHz with PC, too.
</item 1>

<item 2>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The panel has a wider color gamut (110% DCI-P3, for the visual obsessives) and the beautiful, inky blacks that OLED dorks like myself adore.

Steam Deck OLED’s bigger, better battery Valve’s promise of 30-50% more battery raised one obvious question: Is that comparing the OLED model with a Steam Deck
running its original firmware (which had poor battery life), or a Steam Deck using the latest firmware? Valve’s answer: the latter! So if you’re a Steam Deck
owner, expect those big battery improvements over your current Steam Deck experience.

The OLED panel also has a 90 Hz refresh rate — perfect for frame doubling from 45 frames per second. In case that jargon means nothing to you, suffice it to say
that gameplay on the Steam Deck OLED looks smoother.
</item 2>

<item 3>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Manufacturers typically use die-shrinks to either improve performance or efficiency, and Valve has chosen the latter. The max frequency of both the GPU and CPU
are identical between the LCD and OLED models, so for developers tuning their games for Steam Decks, they only have a single performance target.

In concert with the more-efficient processor, there’s also a 25 percent larger battery with a 50Whr capacity, compared to the original’s 40Whr. According to
Valve, these changes combined result in a 30-50 percent increase in longevity between the two generations — the official battery life claim is "three to 12
hours." In our review, we measured a playtime increase of 62 percent when playing Hades on the OLED model versus our 18-month-old original. (Batteries degrade
over time, which explains why our gains were more pronounced.)

The new internals make the Steam Deck OLED marginally lighter, and also cooler to the touch when playing games, aided by a slightly larger fan.
</item 3>

<item 4>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
When asked about the potential for future hardware — like a return of the Steam Machine or an updated Steam Controller — there were no firm updates, but there
does seem to be something happening. Griffais agreed that enabling the Steam Deck to work as a hybrid living room console was a "side goal" of the team, and
noted that everything Valve does to improve SteamOS will benefit other devices. Coomer added that Valve is still working internally on "a bunch of controllers
and prototypes." If SteamOS is being used on larger displays, the team would want to augment that experience with better input options. "We don't have any
announcements on that front, but it's why we're building prototypes and thinking about the future there," Coomer said.

The Steam Deck OLED will be available to order on November 16th at 1PM ET. Valve expects units to begin shipping as soon as orders start to roll in. The updated
pricing on the existing LCD models is already in effect.
</item 4>

<item 5>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
Your time — and money — is precious. Polygon’s What to Buy has rounded up the best in gaming hardware, entertainment, tabletop, merch, and toys, and how to get
them for the best value, so you spend less time shopping and more time enjoying.

Finding the best controller for playing PC games is a more personal journey than it sounds. It’s got to be comfortable to use, with every button, stick, and
trigger easily in reach. It’s also important that you’re getting the features you want. For example, those might include easy setup on PC, good battery life,
multi-platform compatibility, and accessibility options, like the ability to remap buttons and adjust stick or trigger sensitivity.

At Polygon, we play games on a lot of consoles and platforms — both new and old. Our staff has a wide variety of preferences and needs. We’ve come together to
share our favorite PC-ready controllers.

Microsoft Xbox controller

On the Polygon team, the Xbox controller got the highest number of endorsements of any controller.
</item 5>

<item 6>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
For the first year in recent memory, scale did not necessitate tradition, and scope did not preclude getting weird. In 2023, nothing was sacred in video games,
and so they felt more vibrant than ever.

Sure, some of the more “focused” games threw us for a pleasant loop: Dredge begins as a lonely fishing sim before transforming into something otherworldly, and
Humanity morphed from a pensive art project into an all-out war. Dave the Diver, similarly, is not so much about being a diver as it is about running a sushi
restaurant, or hunting for alien artifacts, or conversing with said aliens, or — you get the point. Whether you booted up your Steam Deck for a cross-country
flight or hid your Switch off screen during that boring Zoom meeting, the game you returned to was rarely the one you left behind.

This amorphousness (I’m begging our copy editor to let this one slide, because what other “word” could adequately summarize the video games of 2023?)
</item 6>

<item 7>
The Verge’s 2023 holiday gift guide for kids (The Verge, 2023-12-18T23:14:07+00:00)
Price: $119.99 Amazon Best Buy

8BitDo SN30 Pro (Special Edition) The 8BitDo SN30 is more than a small controller best suited for tiny hands. It also oozes nostalgic vibes to make you feel a
bit of what mom and dad did while playing the original SNES — albeit, with modern games on PC or the Nintendo Switch. What’s more, its see-through color schemes
are sure to entice just about everyone, regardless of what year you were born.

Price: $44.99+ Amazon Walmart 8BitDo

Nintendo Switch Lite (Isabelle’s Aloha Edition) The Steam Deck may have ushered in a new era of handheld gaming, but it’s a bit pricey and overbearing for a
child. That said, there’s still no substitute for the simplicity of Nintendo’s Switch Lite and its travel-friendly 5.5-inch display.
</item 7>

<item 8>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hands on with the Pixel 8 Pro: Darrell took the newly unveiled Pixel 8 Pro for a whirl, and he liked what he saw. While very similar to last year’s model (the
Pixel 7 Pro), Darrell felt that the improved cameras, brighter screen and enhanced AI-powered features made it enough of an upgrade to (potentially) warrant a
purchase — minus the underutilized temperature sensor. Stay tuned for his full review.

Turmoil at Flexport: Dave Clark, the former Amazon executive who was ousted as CEO of Flexport just a year into the job, fired back at its founder and board,
calling recent reporting on the logistics company “deeply concerning.” Clark made the comments Monday in a lengthy post on social media site X following a
report from CNBC that provided new information about his last days at Flexport, a freight forwarding and customs brokerage startup valued at $8 billion.
</item 8>

<item 9>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
(Parts will once again be sold through iFixit.) However, I did enjoy the new power supply, with its faster charging (45 minutes gets you from 20% to 80%) and
2.5-meter cable that stretches the length of my couch.

One small, charming detail: The carrying case now has a tinier carrying case within it. If you want to be extra cautious, stick to the original bulky case. But
if you need to toss the Steam Deck in your backpack for a day trip? Go with the slim case that has funny little thumbstick nipples!

All of the tweaks, large and small, amount to a perfect handheld — made better by the price, which remains the same as the original Steam Deck available today,
just that the OLED will come with more stuff.
</item 9>

<item 10>
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
</item 10>

<item 11>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Like everything else in the Flower Kingdom, a few enemies shift and change in bizarre ways when the Wonder Flower is activated, like Hoppo, the bouncy, round
hippo equivalent that grows in size to create chaos in one of Super Mario Bros. Wonder’s many mesmerizing levels.

All of this weirdness, combined with Nintendo’s platforming expertise, makes Super Mario Bros. Wonder an entrancing romp alone, with friends, or with strangers
online. This is the freshest Mario has felt in decades, and it bodes well for the iconic plumber’s 2D future. —N. Carpenter

3. Alan Wake 2

Developer: Remedy Entertainment

Where to play: PlayStation 5, Windows PC, and Xbox Series X

There’s a body in the forest with a hole in its chest. The forest, like the body, is also missing its heart, and a woman must find both. In each, there are
pages. This is a story, broken into pieces, we learn.
</item 11>

<item 12>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
Sony DualSense controller

There’s another slice of our staff who swear by Sony’s DualSense, a modern controller that retains the symmetrical analog stick arrangement that PlayStation
controllers are known for. Compared directly to the Xbox controller, the DualSense has similar levels of compatibility (both are compatible with Android and
iOS, and feature built-in Bluetooth for support on PC). However, the DualSense does have key differences, as well as a few perks that help to justify its higher
$69.99 price.

In terms of using the DualSense on a PC, it’s possible to do, but it’s a little complicated. It can connect via Bluetooth for a wireless connection, but games
that don’t support the DirectInput controller API likely won’t recognize it as a controller. Plugging the controller into your PC might solve that issue, but
there are other ways to get around this barrier if playing wirelessly is important to you.

One is to rely on Steam, which supports the DualSense controller with its SteamInput API.
</item 12>

<item 13>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
It’s less top-heavy, and it does technically weigh less. These small adjustments have made a noticeable difference during my playtime, and the device feels
slightly more manageable as a handheld — or maybe I’m just too distracted by all of its bright, pretty colors to care about the cramping in my palms.

The thumbsticks on the new handheld have an improved texture that shouldn’t turn gray after a few months’ use, and they feel perfectly serviceable. In my
hands-on time so far, I've found that the OLED touchscreen responds immediately to input, and the haptic feedback on the built-in trackpads feels even more
precise, populating in reactive pin-pricks under the textured plastic.

I spent a few cozy nights on my couch swapping between the LCD and OLED Steam Decks, and honestly, after I’m done with this review, I’ll probably retire my
original model. The Steam Deck OLED is sharper and more responsive; indie games, AAA games and plain old Steam menus look so much better on the new hardware.
</item 13>

<item 14>
Microsoft Activision: What does deal mean for gamers? (BBC News - Technology, 2023-10-13T23:01:24+00:00)
Meanwhile Sophie, who was named Ubisoft's content creator of the year, told the BBC she "just wants the games to win".

"We want everyone to be able to play video games that they want to play, we want them to be accessible for everyone," she said.

"Obviously, we're going to have our opinions, but if the end result is good games, that's fine with us."

This Instagram post cannot be displayed in your browser. Please enable Javascript or try a different browser. View original content on Instagram The BBC is not
responsible for the content of external sites. Skip instagram post by cadaea Allow Instagram content? This article contains content provided by Instagram. We
ask for your permission before anything is loaded, as they may be using cookies and other technologies. You may want to read Meta’s Instagram cookie policy,
external and privacy policy, external before accepting. To view this content choose ‘accept and continue’. Accept and continue The BBC is not responsible for
the content of external sites.
</item 14>

<item 15>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
The PDP Victrix BFG Wireless Controller for PlayStation 5, PS4, and PC in its Call of Duty: Modern Warfare II-branded black-and-gold design is on sale at Amazon
for $169.99 ($30 off). I know this controller looks very Extra (and it is), but it’s a very versatile DualSense alternative if you like FPS and fighting games.
It’s got multi-stage hair triggers, and its sticks and face buttons can be swapped out for an Xbox-style layout or as a six-button fightpad.

Verge Deals on X (formerly known as Twitter) / Join more than 51,000 followers and keep up with the best daily tech deals with @vergedeals Follow us!

Wireless gaming headsets, Laptops, and PC accessories

Asus ROG Flow Z13 $ 893 $ 1300 31 % off $ 893 $ 893 $ 1300 31 % off The ROG Flow Z13 is a gaming tablet with a detachable keyboard, kickstand, and plenty of
chops for QHD gaming.
</item 15>

<item 16>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
One is to rely on Steam, which supports the DualSense controller with its SteamInput API. That makes Sony’s controller a first-class citizen in the eyes of
Steam, right next to the Xbox controller, and plenty of games should offer no-fuss DualSense support.

If you have PC games that you want to play on other services, like PC Game Pass, Epic Games Store, and more, you might experience some middling compatibility
with the DualSense. However, adding them as a “non-Steam game” within Steam can solve for this. To do this, click the “Add a Game” option in the bottom left
corner of the Steam client, then find the executable file (.exe) in your storage.
</item 16>

<item 17>
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
</item 17>

<item 18>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to carve out more of it) that lets Valve’s new
handheld truly deliver on its promise of a portable gaming PC.

All the tiny, delightful improvements of Steam Deck OLED

Since the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will continue with
both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s upgrades. The
original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.

Some changes — even smallish ones — demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new
iteration.
</item 18>

<item 19>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It really shouldn’t work; I can’t imagine another game where all these disparate ideas coalesce so seamlessly. But Dave the Diver would feel less complete
without any one of them. It makes for such a compelling loop, and a consistent advancement of the game’s story, that I kept finding myself in that “one more
day” mindset, eager to jump back into the ocean for one more go. —N. Carpenter

Related Management sim Dave the Diver is a delicate balancing act of absurdity and silliness

11. Resident Evil 4 Remake

Developer: Capcom

Where to play: PlayStation 4, PlayStation 5, Windows PC, and Xbox Series X

It turns out, Capcom is good at remaking games.

The original Resident Evil remake all but set the bar for the format in 2002, with sleeker controls, more nuanced graphical details, and whole new areas to
explore in the iconic Spencer Mansion. The Resident Evil 2 remake changed the entire perspective of its source material without sacrificing the focus on horror
and survival.
</item 19>

<item 20>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
As we donned Samus’ suit and explored strange planets, aggressive alien lifeforms could now get right in our faces, forcing us to dodge, strafe, and roll (in
morph ball form, naturally) using all three dimensions. No longer would we sit back and watch as Samus dipped her toe into a pool of lava; in first-person, as
molten fire spread over our visor, we’d really feel the pressure to find that Varia Suit upgrade. And perhaps most importantly, from behind Samus’ visor, we
gained the ability to scan our enemies and environment, collecting and translating logs from the long-dead Chozo aliens who once inhabited these now-hostile
places.

The world of Prime is harsh and unrelenting. (Save points will, at times, be quite far from one another.) But it’s worth buckling down and pushing through the
pain points to discover this world’s secrets. —M. Myers

18.
</item 20>

<item 21>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
Email (required) Sign up By submitting your email, you agree to our Terms and Privacy Notice . This site is protected by reCAPTCHA and the Google Privacy Policy
and Terms of Service apply.
</item 21>

<item 22>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Both Epic and Google spent a long, long time on subtle lines of questioning. You really had to read between the lines to see that Epic was trying to make a
point about how Google’s 70/30 revenue split is probably based on an arbitrary decision Valve made two decades ago with Steam or how Google was trying to make a
point that Epic, too, likely believed that an app store provides more value than just payment processing and maybe deserves more money.
</item 22>

<item 23>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
And if you just enjoy a good yarn and have access to basically any screen and $15, then you’re a perfect mark too. It runs as well on console and PC as it does
on iOS and Android, so don’t fret about where you play, just do so and soon! Before Square Enix stops investing in all these oddities. —Chris Plante

22. Marvel’s Midnight Suns

Developer: Firaxis Games

Where to play: Nintendo Switch, PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

[Ed. note: Marvel’s Midnight Suns was released in 2022, but it just barely missed the cutoff for our best video games of 2022 list, so it’s eligible for our
2023 awards.]

I know what you’re thinking: Another licensed Marvel game? Come on, right? But hear me out. I played Marvel’s Avengers, too, and this isn’t that.
</item 23>

<item 24>
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
</item 24>

<item 25>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Galaxy-spanning faction quests give you plenty to do, but it’s the side quests that really make you feel like the captain in something Star Trek adjacent:
generation ships and superhero hideouts and colonies of clones. There’s good sci-fi here for those willing to make the journey.

It’s true that the game oversold the idea of a galaxy with over 1,000 planets, a number that pales in comparison to the game’s most obvious competitor, No Man’s
Sky. It’s a shame this became so much of the focus thanks to Bethesda’s marketing, because that same focus obscured the absolutely fascinating new game plus
mode at the game’s heart. Consider this: No Man’s Sky is the game that realizes the universe’s infinite number of planets, while Starfield realizes the
universe’s infinite number of possibilities. —Clayton Ashley

33.
</item 25>

<item 26>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$400 at Amazon (refurbished)

Jsaux Mod Case for Steam Deck $ 24 $ 30 20 % off $ 24 $ 24 $ 30 20 % off A protective modular case system for Valve’s Steam Deck. It consists of a base case
with grippy texture, a built-in kickstand, a removable lid, and a rear mounting system that can attach a power bank using an adjustable strap or Jsaux’s add-on
cooling fan. $24 at Amazon

Jsaux’s 45W USB-C charger with a five-foot cable and included right-angle adapter may be designed for the Steam Deck, but it’s also a great charger for a
Nintendo Switch or just about any USB-C phone or tablet. It’s just $15.19 ($3 off) right now at Amazon.

with a five-foot cable and included right-angle adapter may be designed for the Steam Deck, but it’s also a great charger for a Nintendo Switch or just about
any USB-C phone or tablet.
</item 26>

<item 27>
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
</item 27>

<item 28>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
We know that because they came to us and told us they were doing that. And we know it because after all that went down, I talked to an Activision employee who
was close to that effort, working with the team building the store, and he reported that the entire store team Activision had employed to build their competing
store had been disbanded as soon as Activision signed their deal with Google.

[Editor’s note: We did not hear from such a person at trial; I’d be eager to speak to them now!]

We know Activision was telling Google they were building their computing store; we know in Google internal discussions they said they didn’t want Activision
building a competing store. They agreed to sign this deal, and they were gleeful about the fact they dissuaded perhaps their number one competitor at the time
from launching their own store.

Four years later, Microsoft acquired Activision Blizzard — and one of the big talking points from Microsoft to the European Union was that the merger
strengthened the company overall in order to provide a viable computing store on iOS and Android.
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
</item 30>

<item 31>
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
</item 31>

<item 32>
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
</item 32>

<item 33>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$893 at Amazon

Sony InZone H3 $ 58 $ 100 42 % off $ 58 $ 58 $ 100 42 % off The InZone H3 is the wired (3.5mm / USB-A) model in Sony’s latest lineup of gaming headsets, which
are compatible with both PlayStation and PC. You can customize the audio profile of InZone headsets with the InZone companion app and Sony’s Spatial Sound
Personalizer. $58 at Amazon$60 at Best Buy

Elgato Stream Deck Mini $ 59 $ 80 26 % off $ 59 $ 59 $ 80 26 % off Elgato’s Stream Deck Mini is a smaller, six-key version of its popular macro pads. Its LCD
keys can be customized to offer quick one-button macros during a livestream, or it can simply control shortcuts on your computer and connect to your smart home.
</item 33>

<item 34>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
For developers, this means the device has the same performance targets as the original and there’s no need to create games specifically for the new hardware. In
the end, the device weighs about 30 grams less than the original model, or the equivalent of five quarters.

Engadget

In use

Which leads us to the bad news for my tiny hand gang. The Steam Deck OLED is the same monstrous size as the original: 11.7 inches long, 4.5 inches tall and
nearly 2 inches thick. The thumbsticks on the new model are actually 1mm taller than the first version. This was one of my main complaints about the Steam Deck
when I first reviewed it: I have smaller-than-average hands, and the Steam Deck looks and feels absolutely ridiculous when I’m holding it. It’s just massive, no
matter how long my manicure gets. However, the OLED model feels more balanced than the original. It’s less top-heavy, and it does technically weigh less.
</item 34>

<item 35>
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
</item 35>

<item 36>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
You know, we expanded the company to nine times the size we were before Fortnite took off. And so we’re just trying to forge the strategy where we’re living
within our means and doing everything we absolutely can.

But we’ve gotten so much traction with Fortnite in so many different ways, and with the Epic Games Store, right, we’re going to continue investing heavily here.
The Epic Games Store is the untold success story in the background. Since it launched in 2018, we now have 80 million monthly active users. Steam has 120
million, so we’re catching them fast! Fortnite has more concurrent users right now than all Steam games combined. You should really expect dramatic changes that
benefit all developers over the next few years.

[Editor’s note: Sweeney isn’t right about Fortnite having more concurrent users than Steam. Epic spokesperson Natalie Muñoz confirmed Sweeney misinterpreted a
Polygon story about how Fortnite hit 7.6 million players at a time when the entirety of Steam had 10.1 million.
</item 36>

<item 37>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Other improvements include support for both Wi-Fi 6E and Bluetooth 5.3, together with a new antenna array that will give a better experience when using both at
the same time. There’s also a reworked power supply (it has a logo now!) with a longer 2.5m cable that will charge the OLED model from 20 to 80 percent in “as
little as 45 minutes.” The thumbsticks have also been made 1mm taller, and the touch pads' haptics are slightly sharper.

Valve

The Steam Deck OLED starts at $549, and will replace all but the entry-level model in the range.
</item 37>

<item 38>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
This is a story, broken into pieces, we learn. One that ended badly before. One that will likely end badly again. Will you still put it together?

Alan Wake 2 is full of words but few are in order. It is set in places of confused geography, full of people who don’t quite belong, in rooms that might be
different each time you enter them. Alan Wake 2 is wrong, and you must make it right, if you can.

Remedy Entertainment’s assured masterpiece is less about stories and more about dreams — the way they can slip from frightening to absurd at a moment’s notice,
about how we can get lost in them, learning from our experiences or succumbing to insecurities. How video games can mimic their shape, or lack thereof.

It is about repetition, and the way we find meaning in stories and people and places by returning to them over and over, wondering if we changed or they did.
</item 38>

<item 39>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
Also, the PS5’s interface for the Edge is intuitive, and it makes remapping controls and setting schemes the breeze that it should be.

While the Edge is compatible on PC, Sony and Steam haven’t developed drivers that let gamers take full advantage of its back paddles and swappable control
schemes.

Nintendo Switch Pro controller

The Switch Pro controller is another staff favorite. If it’s a great Nintendo Switch and PC controller you’re after, this is the pinnacle in terms of comfort,
durability, battery life, and other Switch-specific extras, but we’ll get to those a little later.

The Switch Pro controller can be used to play PC games. But like the DualSense, it’s not a plug-and-play affair unless you literally plug it in via USB. To use
it wirelessly, running Steam can help you out, as it officially supports the Switch Pro controller. Adding non-Steam games to your Steam library is a clever way
to ensure compatibility for games that support controllers.
</item 39>

<item 40>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$59 at Amazon

Storage for consoles and PCs

A massive 1TB SanDisk Extreme microSD card is now $89.99 $79.99 ($13 off) at Amazon. That can store a whole lot of games on a Steam Deck or Nintendo Switch.

Samsung Evo Select microSD card (512GB) $ 30 $ 40 25 % off $ 30 $ 30 $ 40 25 % off This Samsung Evo Select microSD card offers 512GB of extra storage for a
Nintendo Switch, Steam Deck, or Asus ROG Ally at a good price point. $30 at Amazon

Western Digital WD_Black SN850X SSD $ 70 $ 180 61 % off $ 70 $ 70 $ 180 61 % off The WD_Black SN850X is one of the fastest M.2 drives around, with a sequential
read speed of up to 7,300 MB/s and a write speed of up to 6,600 MB/s.
</item 40>

<item 41>
Teenage Engineering's K.O. II sampler proves the company can do cost-friendly cool (Engadget, 2023-12-19T15:00:38+00:00)
Before our review unit even showed up, several users were complaining that theirs had defective faders. Enough folks were having this problem that it quickly
became known as “fadergate.” One brave creator even took their unit apart and, possibly, discovered the cause — the internal pins were bent and not making a
connection. I asked Teenage Engineering about the issue and will update this story once I hear back.

Some buyer’s theorized that the issue might be caused by the fact that the K.O. II ships without the caps on its rotaries and fader. They come loose in the box
to enable the packaging to be flatter, but the cap for the fader is unusually tight fitting. This led to speculation that the enthusiasm required to push this
cap down might be putting too much pressure on components inside, opening up the cruel possibility of users breaking their own devices before they even got to
play with them.
</item 41>

<item 42>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
Update (Nov. 17): Checked prices and links for accuracy.
</item 42>

<item 43>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Engadget

Even with improvements to the battery, thermals and weight distribution, the Steam Deck OLED is still more of a Wii U than a Vita. It’s a beast of a handheld,
and it makes the most sense as a portable console to use around the house, rather than a truly travel-ready device. The new model even comes with an extra-long,
2.5m charging cable, designed specifically with couch-based players in mind. I didn’t test out the docking station with the new Steam Deck, but Valve recently
released a SteamOS update that lets players take advantage of HDR and VRR on external displays, which only drives home the Wii U-ness of it all.

As far as software, you know the drill by now. Valve has rolled out labels for Steam games signifying how well they’ll work on the handheld, with four levels:
unknown, unsupported, playable and verified. Playable games generally come with a small caveat, like having to deploy the on-screen keyboard, while verified
games have been fully optimized for portable play.
</item 43>

<item 44>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
However, its biggest benefit is its Gulikit Hall effect analog sticks, which are resistant to developing drift over time by design. By comparison, the
potentiometer-based sticks used in the Ultimate 2.4g (and all of the other controllers mentioned so far) bear no such guarantee. There’s a possibility you may
never experience the issue firsthand, but the anti-drifting sticks alone might be a reason for some to consider buying the Ultimate Bluetooth.

The Ultimate Bluetooth is similar in a lot of ways to Nintendo’s Switch Pro controller, with its motion control support, rechargeable battery, and
Nintendo-style face button layout. However, it doesn’t have all of the same features. For instance, it doesn’t support NFC, so you can’t tap an amiibo on it,
and its haptics aren’t up to par with Nintendo’s. However, it’s one of a few third-party Switch controllers that can wake your console from sleep. If it’s
paired, press the home button, then shake the controller to awaken the Switch.

Update (Nov.
</item 44>

<item 45>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
That’s not bad considering Google’s first-gen smartwatch retailed for $349.99 when it first launched a little over a year ago.

The first-gen Pixel Watch may not be as capable as the new Google Pixel Watch 2, but they have a number of strengths in common. Both offer health and
fitness-tracking features powered by Fitbit, allowing you to enjoy special features like FDA-cleared EKGs. The wearables also support Google services like
Google Assistant and Google Wallet. And now, thanks to the new Wear OS 4 update, you get access to the Google Calendar app, Gmail, and the new Safety Check
feature introduced with the new wearable.

There are some trade-offs, though. The Pixel Watch 2 offers superior battery life, whereas the last-gen model struggles to last a full day. The newer wearable
also sports some new tricks, including Sleep Profiles, abnormal heart rate tracking, and nightly SpO2 features.
</item 45>

<item 46>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
GPU updates were the centerpiece of the news for several reasons. One is a feature the company calls Dynamic Caching. Apple writes,

With Dynamic Caching, only the exact amount of memory needed is used for each task. This is an industry first, transparent to developers, and the cornerstone of
the new GPU architecture. It dramatically increases the average utilization of the GPU, which significantly increases performance for the most demanding pro
apps and games.

More simply put, the idea is allocating just the right amount of local memory to a given task. It’s something Apple has been working on for a while. The other
two additions should be more familiar for anyone who’s spent time behind a game controller. The first is ray tracing, a feature designed to bring added realism
to the gaming experience, though more accurate depictions of lighting and reflections.

That feature is now hardware-accelerated, along with mesh shading. The triangle computer shader makes processing complex geometrical images more efficient.
</item 46>

<item 47>
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
</item 47>

<item 48>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The triangle computer shader makes processing complex geometrical images more efficient. That, in turn, allows for higher-quality graphics and more scene
detail.

The results are impressive. I’ve been spending a bit of time with the Soulslike title, Lies of P. It’s basically Pinocchio with more swords. It’s of a small –
but increasing – number of day-and-date releases for macOS. A couple of years ago, it would have been inconceivable for the operating system to get the game the
same day as PS5, Xbox Series X/S and Windows. That said, the macOS gaming library still has a long way to go.

Indeed, the gameplay was smooth and detailed – something that also would have seemed impossible not all that long ago. Sony’s PS5 DualShock controller connects
seamlessly and makes for a fun portable gaming experience.

One thing I noticed during last week’s presentation is that Apple is leaning less heavily on Intel comparisons.
</item 48>

<item 49>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 49>

<item 50>
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
</item 50>

<item 51>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
“The Xbox wireless controller just works for me — it fits right in my hands, the texture feels just right, and it’s not too heavy” —Nicole Carpenter

Beyond PC compatibility, the Xbox controller boasts native compatibility on Steam Deck, plus recent versions of Android and iOS. So, whether you have a phone or
tablet, this controller can easily connect via Bluetooth. A growing number of mobile games support it, too, and it works like a charm with the Xbox Game Pass
game streaming app.

It can work on even more platforms if you purchase 8BitDo’s clever $19.99 USB Adapter 2. With it, you can use the Xbox controller on your Nintendo Switch. In
addition to Switch support, this USB adapter lets you easily connect your Xbox controller to other devices, like Mac, Raspberry Pi, and more.

The biggest downside to the Xbox controller might not bother you: It requires two AA batteries, though on a positive note, the battery life is around the
40-hour range, depending on usage.
</item 51>

<item 52>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
There are so many puzzles in this game — there’s even a puzzle metagame, spread across the game’s gorgeous map — and they’re each excellent, teaching you new
concepts before refracting them and forcing you to think differently. —Nicole Clark

Related November games you might have missed

44. Fading Afternoon

Developer: yeo

Where to play: Windows PC

Video games demand an account for your time. Some video games track the minutes and hours you’ve spent on screen. Others impose strict limits on how long you
play, or how much you can do. But all video games constantly ask: How will you spend your time? Seiji Maruyama doesn’t know how much time he has left. Fresh out
of prison, he’s not a young man anymore, but he doesn’t want the streets to know. So he returns to his life as a yakuza heavy, hoping to make his mark again.

Fading Afternoon ends when Seiji’s time runs out.
</item 52>

<item 53>
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
</item 53>

<item 54>
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
</item 54>

<item 55>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Playing the Steam Decks back-to-back highlighted another significant quality-of-life improvement: heat management. My palms have gotten toasty playing the OLED
model, but they haven’t broken a sweat like they often do with the original. The new fan may be larger, but it’s not any louder, and it clearly works a treat.

The Steam Deck OLED has a new Wi-Fi 6E module and improved antennas that should enable faster downloads for anyone with a compatible router. I don’t have a 6E
router or multi-gigabit Wi-Fi, but with a regular Wi-Fi 6 router, the top download speed I saw on the new Steam Deck was 562mbps, which was about 10 percent
faster than the old Steam Deck, and 10 percent slower than the speeds I get on my MacBook Pro. The device uses Bluetooth 5.3 and adds a dedicated antenna, which
should improve functionality when using Wi-Fi and Bluetooth at the same time.
</item 55>

<item 56>
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
</item 56>

<item 57>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
The best deals of the year were once known to occur during Black Friday and Cyber Monday, which in 2023 take place on Friday, Nov. 24, and Monday, Nov. 27,
respectively. Not anymore!

In 2023, the best Black Friday deals kicked off on Nov. 17 at Amazon, Best Buy, Walmart, and other retailers. Instead of hosting all of the doorbuster deals on
one day right after Thanksgiving, retailers see the value in maximizing the amount of time you have to shop for products before Black Friday. And I’ve gotta
say, it’s quite nice to not feel rushed while shopping.

In addition to presenting you with the best board game deals and price drops on gaming monitors, plus deals on SSDs and microSD cards, there are some bits of
info and wisdom I’ve organized into sections below that might help you feel more informed as you begin to shop for the holidays.
</item 57>

<item 58>
Microsoft Activision: What does deal mean for gamers? (BBC News - Technology, 2023-10-13T23:01:24+00:00)
And critically, Microsoft has a habit of making its big blockbuster games available to play on Game Pass, for a monthly fee, on the first day they release.

Image source, EPA-EFE Image caption, Bethesda, which is owned by Microsoft, launched its new game Starfield in 2023 - but only on Xbox and PC

It's up to gamers to decide for themselves whether it's a better deal to buy the game or pay to access it.

But it is worth noting this concerned the UK regulator so much it made sure Activision Blizzard games couldn't be streamed exclusively on Xbox for 15 years.

That might be because Microsoft has form here - when the hotly anticipated Starfield came out in September 2023 to critical acclaim, it was available
exclusively on PC and Xbox, and playable on Game Pass from day one.

If games are still on PlayStation, what's in it for Microsoft?

Even with games still able to appear on PlayStation, it's Microsoft who will be winning out.
</item 58>

<item 59>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The immediate reality is a bit of a mess and a little complicated, but this, Valve says, will be the Steam Deck line-up moving forward:

256GB Steam Deck LCD ($399)

512GB Steam Deck OLED ($549)

1TB Steam Deck OLED ($649)

As a result of the changes, the 64GB LCD Steam Deck (previously sold at $399) will drop to $349, and the 512GB LCD (previously $649) will drop to $499. Once
Valve sells through its inventory of both, those lines will be discontinued. Valve says it will continue to support all LCD models with software updates.

So, the long-term lineup starts at the same $399 price, which gets you the old LCD Steam Deck, but with 256GB of storage instead of 64GB. The new step-up model
costs $20 more at $549, but comes with an OLED display and the improvements outlined above, plus double the storage.
</item 59>

<item 60>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—CA

Related Viewfinder is puzzle game heaven

24. Humanity

Developer: tha

Where to play: PlayStation 4, PlayStation 5, and Windows PC

Humanity looks like a modern art project, recalls memories of early PlayStation oddity, and comes (in part) from the team behind game of the future Rez
Infinite. But its most impressive feat is the way it gradually shifts genres, starting as a classic puzzle game, then taking on tower defense and shoot-’em-up
traits as it evolves into an all-out war. Think Braveheart, with a shiba inu general leading a troop of faceless, brainless low-polygon figures against a group
of angry, lightsaber-wielding, even lower-polygon foes. It’s the kind of design subtlety that stays in its lane yet builds on you, and before you realize it,
ends very differently than it began. Much like humanity itself? —ML

23.
</item 60>

<item 61>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
It sports drift-free Hall effect sticks and is compatible with PC, Nintendo Switch, and Android.

8BitDo Ultimate Bluetooth Controller $ 56 $ 70 20 % off $ 56 $ 56 $ 70 20 % off The 8BitDo Ultimate Bluetooth Controller comes with its own charging dock,
features remappable controls, and is compatible with the Nintendo Switch, Steam Deck, and Windows PCs thanks to its Bluetooth and 2.4GHz connectivity. Unlike
the 2.4GHz-only version, it features drift-free Hall effect sticks. $56 at Amazon (black)

The RGB-ified PowerA Spectra Infinity Enhanced Wired Controller for Xbox and PC is on sale at Amazon for $29.99 ($15 off). In addition to its colorful lights,
it’s one of the cheapest controllers around to sport hair trigger lockouts.

for Xbox and PC is on sale at Amazon for $29.99 ($15 off). In addition to its colorful lights, it’s one of the cheapest controllers around to sport hair trigger
lockouts.
</item 61>

<item 62>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
There’s no better controller for the PS5 than this one. It’s a good controller for using PS Remote Play, too. When it’s connected to an Android device for a PS
Remote Play session, the DualSense supports adaptive triggers in games that take advantage of the feature.

8BitDo’s $19.99 USB Adapter 2 also works perfectly with the DualSense, letting you play games on your Switch with it once it’s paired. I preferred to play
Splatoon 3 with my PS5 controller’s analog stick arrangement. Heck, it even supports gyroscopic camera movement, just like the official Switch controllers.

Just like Microsoft, Sony also has its own pro-grade controller. It’s the $199.99 DualSense Edge, which is compatible with the PS5. It’s tempting to call out
its similarities to the Xbox Elite Series 2, of which there are plenty, but it’s more helpful to speak about how it differs. For one, its stick modules can be
replaced entirely — a move that makes the DualSense Edge a little more user-repairable.
</item 62>

<item 63>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
Again, you can do that by clicking the “Add a Game” option on the bottom left corner in the Steam client.

“This is a very solid, unassuming but beautifully engineered pad, with Nintendo’s typically great ergonomics and switchgear. The D-pad is one of the best out
there, and those lovely, fat, responsive face buttons will give you SNES flashbacks in a good way” —Oli Welsh

Alternatively, if you don’t want to use Steam to play your games, relying on DS4Windows and its bevy of drivers is a free yet slightly advanced way to make
Windows recognize your Switch Pro wireless controller. Of course, that $19.99 8BitDo USB Adapter 2 that I keep referencing can come in handy here, too. After
plugging it into your PC and tethering wirelessly to it, you’re good to go — no fussing with apps required.

The Switch Pro controller is, unsurprisingly, a fantastic Switch controller.
</item 63>

<item 64>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is the highlight of the device, while things like faster charging, improved antennas and smoother haptics are welcome bonuses. More
importantly, this is Valve renewing its commitment to portable PC gaming, and that’s a relief to see. The Steam Deck is the granddaddy of handheld PCs and Valve
has the resources to continue innovating in this space. For now, the OLED model is a half-step toward the Steam Deck 2, which may be the final version of the
device — unless Valve finally learns how to count to three.
</item 64>

<item 65>
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
</item 65>

<item 66>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
As the dangers around you grow, so too does your ship’s capabilities. And by adventuring into battles — metaphorical and otherwise — with the seas’ most
dastardly critters, you’ll always come out on the other side with some upgrades that allow you to catch even better fish and build an even bigger boat.

By the time you’ve spent 10 or so hours with Dredge, you’ll feel like a commercial fisherman who just happened upon something bigger and more foreboding than
they could have imagined — and that, perhaps, you’ve stared deep into some kind of black abyss, only to escape forever changed. —RG

Related Dredge invents and perfects the fishing horror genre

16. Final Fantasy 16

Developer: Square Enix

Where to play: PlayStation 5

Final Fantasy 16 kicks ass. The newest mainline entry in the long, winding series takes you on a lavish, unadulterated Game of Thrones-esque adventure.
</item 66>

<item 67>
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
</item 67>

<item 68>
The best portable chargers for your gear (Polygon, 2023-11-17T13:10:12+00:00)
If you buy something from a Polygon link, Vox Media may earn a commission. See our ethics statement .

Your time — and money — is precious. Polygon’s What to Buy has rounded up the best in gaming hardware, entertainment, tabletop, merch, and toys, and how to get
them for the best value, so you spend less time shopping and more time enjoying.

A power bank, also commonly referred to as a portable charger, is an essential purchase for anyone who’s on the go with a phone, tablet, or a gaming handheld
like a Nintendo Switch or a Steam Deck. It lets you recharge your precious gadgets one or more times via a USB connection, so you can keep talking, texting, or
gaming throughout the day.

Power banks range in price depending on a few factors: how compact it is, what kind of ports it has, how much battery capacity it contains (typically measured
in mAh), and how quickly it can recharge your devices.
</item 68>

<item 69>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
Google Pixel Tablet

Photo by Cherlynn Low / Engadget Google Pixel Tablet $409 $499 Save $90 The 11-inch Pixel Tablet comes with a charging speaker dock that allows it to double as
a smart home hub. This deal marks a new all-time low. $409 at Amazon

The 128GB Google Pixel Tablet is down to $409, which is $90 off and a new low. The 256GB model is also on sale for $509. While we think Samsung’s Galaxy Tab S9
works better purely as a tablet, Google’s 11-inch slate still gets you a sharp LCD display, a speedy Tensor G2 chip and an easy-to-hold design that weighs just
over a pound. Its big hook, however, is that it can attach to an included charging speaker dock and double as a Nest Hub Max-style smart display. Android still
isn’t as optimized for large screens as iPadOS, but if that extra functionality sounds useful to you, the Pixel Tablet is worth a look.
</item 69>

<item 70>
Discord’s latest teen safety blitz starts with content filters and automated warnings (Engadget, 2023-10-19T15:00:31+00:00)
For users, we don’t want them to be gated from having these custom experiences at their fingertips,” Cherry Park, director of product marketing at Discord, told
Engadget. “In terms of the way we architecture and build it, there are a couple of solutions. Some are easier, some are more difficult, and you're going to see
us experiment with a few of them over the next few quarters.”

Discord promised to share more about its efforts around app portability in the near future. In the meantime, Nitro subscribers will get a chance to test new
features before they become available to the public.

Update, October 20, 2023, 11:20 AM ET: This story has been updated to note that these feature are rolling out in the coming weeks, not this week as the post
originally said. We also removed a mention that Discord would automatically delete potentially sensitive content from strangers. This was originally part of a
briefing we received on the new features, but it was not included in the final release.
</item 70>

<item 71>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
In October 2019, Epic internally said it might pursue an aggressive pursuit model with the Epic Games Store, where it would aggressively pay more for exclusive
games and really pull in more and more users. Are you in the aggressive pursuit model?

No, we’re in a different model, which is called the “Ridiculously aggressive pursuit model.” If it’s a 10-slide deck, that’s on slide 11.

Did you get a bingo on your Verge bingo card?

Yeah, I think it was almost a complete shutout. I think there was only one cell in the entire board that was missing. I was really impressed — the thing for
that bingo card that really impressed me was “lick the cookie.” Do you remember? You may not even be old enough, but “lick the cookie” was in the 1999 US v.
Microsoft antitrust trial.
</item 71>

<item 72>
Save on the winners from the 2023 Game Awards this weekend (Polygon, 2023-12-09T15:00:00+00:00)
We’re coming up for air from our coverage of The Game Awards to loop our readers in on some of the best deals happening this weekend. If you’re looking to
experience some of the winners from the 2023 Game Awards for yourself, you can find discounts on a variety of titles on Steam. Some other highlights you might
want to check out include a sitewide sale on Blu-Rays from The Criterion Collection, and the Xbox Series X matching its Black Friday price on Walmart and
Amazon.

As we do every weekend here at Polygon, we’ve rounded up a collection of our favorite deals from the worlds of gaming and entertainment, along with some of the
best-selling products that have made a recent appearance on our site.

The best gaming deals this week

Steam is discounting many contenders from the 2023 Game Awards and winners from the past nine years.
</item 72>

<item 73>
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
</item 73>

<item 74>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
In much the same way, the gameplay feels surprisingly modern at a distance, but on closer inspection, you start to see how this proto-immersive sim is actually
what inspired so much modern game design. You have to rely on your own curiosity, caution, and cunning to navigate the halls of Citadel Station and upgrade your
hacker into a cybernetic death machine. It’s a game that expects a lot from the player (and a little save scumming), but the experience is just as rewarding as
it was in 1994. —CA

Related The System Shock remake does something remarkable

32. Star Wars Jedi: Survivor

Developer: Respawn Entertainment

Where to play: PlayStation 5, Windows PC, and Xbox Series X

Is Star Wars Jedi: Survivor an Empire Strikes Back level of sequel? Well, no, but it manages to get extraordinarily close to being one of the best follow-ups in
the entire Star Wars franchise.
</item 74>

<item 75>
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
</item 75>

<item 76>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.
</item 76>

<item 77>
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
</item 77>

<item 78>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Engadget

Hades is my most-played game on the original Steam Deck, so I used that to test out Valve’s battery life claims. Though it doesn’t support HDR, a game like
Hades pops on the OLED, and its animations look more fluid than on the LCD version. The OLED lasted four hours and 23 minutes before dying. With the same
settings (800p with a 60 fps cap and similar brightness) the original model died after two hours and 42 minutes. That’s a 62 percent improvement, surpassing
Valve’s own estimates, though batteries degrade and I’ve had my LCD Steam Deck since the device’s launch in early 2022. In a stress test that involved playing
Elden Ring with everything maxed out, the OLED battery lasted two hours and 20 minutes, about 40 percent longer than the original.

Playing the Steam Decks back-to-back highlighted another significant quality-of-life improvement: heat management.
</item 78>

<item 79>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The original Steam Deck had a 7-inch 800p LCD running at 60Hz. The new model squeezes a 7.4-inch OLED screen into the same space, significantly shrinking the
bezels around the display. The new panel runs at the same resolution, but with a refresh rate of up to 90Hz. It’s also HDR, covering 110 percent of the P3 color
space with a peak brightness of 1,000 nits — the old model maxed out at around 400 nits. Like all OLED screens, it has pure blacks and near-instantaneous
response times. As before, anti-glare etched glass is available, but only on the premium model.

The Steam Deck OLED has a die-shrunk version of the same custom APU found in the original Steam Deck. The new APU is produced on a 6nm process, rather than the
older 7nm process.
</item 79>

<item 80>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Octopath Traveler 2 leaps across that boundary. In place of the original game’s repetitive level design, monotonous narrative structure, and sometimes awkward
characterization, the sequel demonstrates an expert ability to challenge your expectations at every turn. Yes, your general goal is still to recruit eight
playable characters (hence the name) and follow each of their separate plot threads to their respective conclusions, participating in turn-based battles and
side quests along the way. But said plots vary greatly from character to character, and if you so choose, you can see a handful of characters through several
major plot points before recruiting the whole gang. Octopath Traveler 2 finely toes the line between that comfort food-esque repetition of the best JRPGs, and
the subversive nature of great genre storytelling. —M. Mahardy

12. Dave the Diver

Developer: Mintrocket

Where to play: Nintendo Switch and Windows PC

You could describe Dave the Diver as a fishing game and a restaurant management simulator, and that’d be correct.
</item 80>

<item 81>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—Julia Lee

Related September games you might have missed

45. The Talos Principle 2

Developer: Croteam

Where to play: PlayStation 5, Windows PC, and Xbox Series X

Like its predecessor, The Talos Principle 2 tackles grand science fiction ideas, particularly about what it means to be human — a theme thoroughly explored in
this installment of the series set in a post-human society of AI-powered robots that are carrying on human culture and civilization. Also like its predecessor,
The Talos Principle 2 is replete with philosophical commentary and references to famous artists and thinkers. My favorite is a riff on Werner Herzog’s famous
quote about birds: “The enormity of their flat brain. The enormity of their stupidity is just overwhelming.”

But if you’re like me, you’re playing the game because you’re an absolute freak for light refraction, gravity, and geospatial puzzles.
</item 81>

<item 82>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This investment alone means any time you buy a game built on Epic’s Unreal Engine, Tencent (and Sony) is getting a cut. Tencent is the biggest investor in
games, with thousands of tendrils across the industry — if you played something this year, Tencent was probably involved.

On a smaller scale, companies like Netflix and Devolver Digital have also dabbled in acquisitions recently. Devolver started buying studios in 2020, and it now
owns Croteam, Dodge Roll, Doinksoft, Firefly Studios, Nerial and System Era Softworks. Annapurna Interactive bought South African studio 24 Bit Games in
November. Netflix launched its Games division in 2021, and it’s already purchased four studios, including Oxenfree developer Night School and Alphabear company
Spry Fox.

Night School co-founder Sean Krankel told Engadget in June that the move to Netflix was a boon for the studio, providing financial security, a dedicated working
space and plenty of marketing support for its projects.
</item 82>

<item 83>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The HDR OLED model is what they would have released the first time around, if the technology had existed before 2022. The handheld market has evolved quickly —
in just over a year, the Steam Deck proved there’s widespread demand for handheld PCs, and its success helped drive chip makers and display manufacturers to
build hardware specifically for high-fidelity mobile play. This is great news for Valve and the Steam Deck, and also the other handhelds coming out of companies
like Ayaneo, ASUS, Ayn, Logitech, Retroid and Razer. Hell, maybe even Playdate will get an HDR OLED display one day. (Just kidding; Playdate is literally
perfect).

So, no, the HDR OLED refresh isn't the Steam Deck 2, but it is a welcome improvement on an already good device. The Steam Deck OLED is the facetuned,
photoshopped, spit-shined version of Valve’s handheld, and it features tangible, clever improvements.
</item 83>

<item 84>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The gameplay didn’t just help me stick with the game, but instead allowed my excitement to bubble over every time I took on a new mission. —Ana Diaz

15. Cyberpunk 2077: Phantom Liberty

Developer: CD Projekt Red

Where to play: PlayStation 5, Windows PC, and Xbox Series X

Like the best DLC, Phantom Liberty does as much to expand on the base game as it does to reframe it. Alongside the sweeping Patch 2.0, which revamped Cyberpunk
2077’s role-playing progression systems, improved its enemy AI, and course-corrected a litany of other details, Phantom Liberty also adds a whole new district
to the dystopian world of Night City, complete with its own spy-thriller storyline.

The resulting Cyberpunk 2077 is not an entirely different beast than the one that was released in 2020, but it is a more evolved one.
</item 84>

<item 85>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
And so even as Microsoft was crushing corporate competitors, the developer experience was excellent. [Editor’s note: Netscape might feel differently.]

“Even as Microsoft was crushing corporate competitors, the developer experience was excellent.”

Google and Apple both treat developers as adversaries — they try to attack our revenue streams and prevent us from competing with their products. They’ve built
these massive self-preferencing schemes all around excluding developers and disadvantaging third-party developers. I think this is very shortsighted. I think
any tech company — Apple, Google included — would be much better off in the long term if they viewed developers as awesome partners and did everything they
could to support them and empower them and not get in their way financially.

And this has been our philosophy with Unreal Engine, for example, and the Epic Games Store. We just want to be a cool partner that helps other companies succeed
the way we do. And I think philosophy change... perhaps it will only come with a generational change in the company’s management.
</item 85>

<item 86>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.
</item 86>

<item 87>
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
</item 87>

<item 88>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The Steam Deck OLED now includes a Wi-Fi 6E radio (great if you have a modern router and connection to make use of it) and a dedicated Bluetooth module, which
allows for improved audio quality with wireless headphones, the option to wake the Steam Deck from a controller, and the capability to connect many Bluetooth
controllers for multiplayer. The haptics are noticeably improved, with a wider frequency range. And the touchscreen is markedly better — it’s responsive to the
point where I don’t feel the need to plug in a USB-C keyboard anytime I need to type a few words.

A review for spec specifics If you’re the sort of person who likes the nitty-gritty technical details of screens and batteries, we also recommend reading The
Verge’s review.

Valve also claims that the Steam Deck OLED is more repairable, which, mercifully, I didn’t have the need to test for myself. (Parts will once again be sold
through iFixit.)
</item 88>

<item 89>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The first thing to know about the Steam Deck OLED is that it’s not the Steam Deck 2. The Steam Deck OLED is a mid-cycle refresh from Valve, similar to the
Switch OLED upgrade, but there’s a lot more going on internally here. Its screen is better, its battery life is better, its thumbsticks are better, the haptics
are better, and its chip and thermals are better. One thing that hasn’t changed is its form factor — this is still one of the biggest babies on the handheld
block.

The new Steam Deck’s main upgrade is its OLED screen, which replaces the original LCD and, crucially, has native support for HDR and faster frame rates. The
OLED also has smaller bezels, making the display larger at 7.4 inches, compared with the original 7-inch screen.
</item 89>

<item 90>
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
</item 90>

<item 91>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
In early 2022, Valve released my favorite video game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of
my sprawling collection of video games wherever I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much
better games looked on our family’s OLED TV.

And yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.
</item 91>

<item 92>
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
</item 92>

<item 93>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
We strongly recommend OnePlus’ magnetic keyboard, which is also on sale for $100 ($50 off), however, if you purchase directly from the company's website, you
can get the keyboard bundled for free.

The 9th-gen iPad (8/10, WIRED Recommends) is our favorite iPad for most people, even though it's one generation behind. Aside from costing less than the current
10th-gen model, it is still compatible with the same accessories as the first-gen Apple Pencil. It retains the physical Home button with Touch ID at the bottom
of the screen.

Pixel Tablet Photograph: Google

Google's Pixel Tablet (7/10, WIRED Recommends) is part tablet, part smart display. It comes with a charging dock that doubles as a speaker, so you can pump out
the tunes and get answers from Google Assistant at a moment's notice. Take it off the charger and you get a full Android tablet experience.
</item 93>

<item 94>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Yes. But you might find that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year,
the upgrade is inescapable.

And if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.

The Steam Deck OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has
affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .
</item 94>

<item 95>
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
</item 95>

<item 96>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The improved 90 Hz refresh rate falls short of the Asus ROG Ally’s 120 Hz, though it’s worth considering that the higher the frame rate, the faster the battery
drain — assuming you can even spot the difference between 90 Hz and 120 Hz.

You know what’s more interesting than a list of technical specs? Actually seeing the screen. It’s gorgeous. On Sunday morning, I picked at Sega’s Like a Dragon
Gaiden: The Man Who Erased His Name while watching football and Christmas movies with my family. The new Yakuza game looks fantastic on high settings at 60 fps,
its night sequences flaunting the OLED’s dark blacks and the abundance of neon signage showcasing the HDR.

I got so distracted pulverizing bad dudes that it took me some time to notice the Steam Deck OLED’s most welcome feat. When playing AAA games on the original
Steam Deck, I had always kept the handheld plugged into an outlet. But with the OLED model, a couple of hours had passed before I realized I’d forgotten to
connect the charger.
</item 96>

<item 97>
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
</item 97>

<item 98>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
I played Marvel’s Avengers, too, and this isn’t that. It might seem like it’s going to be at first, because Midnight Suns makes the grave error of introducing
Iron Man and Doctor Strange as its tutorial characters, and these two might just be the most irritating characters in the entire video game. (I have beaten the
game, so I am allowed to make this call.) You must press on and give Midnight Suns time to win you over. Because it has so, so much more to offer than it may
appear in its first few hours.

Picture the romance and humor of Fire Emblem: Three Houses, combined with the high-stakes tactical battles of XCOM 2 — that’s what Midnight Suns becomes in its
mid-game and endgame. It’s a card-based strategy game, and each hero has their own customizable deck. I started off favoring Captain Marvel, Magik, and Blade,
simply because their moves and hilarious dialogue kept me entertained, but I soon realized that every single character has something exciting or unexpected to
bring to the battlefront.
</item 98>

<item 99>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—Ana Diaz

49. Lies of P

Developer: Neowiz Games

Where to play: Mac, PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

Yes, Lies of P is a Dark Souls mixed with Pinocchio, and that’s a questionable elevator pitch from the outset.

In the years leading up to Lies of P, “Pinocchiosouls” was more of a running joke than anything — this profane idea that you can take any world and slap some
Dark Souls into it to get people mildly interested. But once you’re in the game, eliminating bosses left and right with your sweet parry moves, you’ll quickly
find yourself entirely unbothered by how strange Lies of P initially seemed. And you’ll start telling your partner things like “I have to go back to Geppetto to
upgrade my puppet body” like it’s a perfectly normal task to assign yourself on a Tuesday afternoon.
</item 99>

<item 100>
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
</item 100>
=== END ITEM id=0364e0516f8f9761 ===
