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

=== ITEM id=1eaa21009070277b ===
Gold sentences and facts:
Sentence 0: "And if the Braves let this be the story of their series, that's on them.
  Fact 0.0: A quoted speaker says that if the Braves let this be the story of their series, that is on the Braves.
Sentence 1: But while Nocturne could have easily settled and coasted on classic Castlevania vibes, the new show steps its game up considerably by using the franchise’s iconic lore to challenge your understanding of what it truly means to fight for freedom, equality, and fraternity.
  Fact 1.0: Nocturne could have easily settled and coasted on classic Castlevania vibes.
  Fact 1.1: The new Nocturne show steps its game up considerably.
  Fact 1.2: The new Nocturne show uses the Castlevania franchise's iconic lore.
  Fact 1.3: The new Nocturne show uses the franchise's iconic lore to challenge the viewer's understanding of what it truly means to fight for freedom, equality, and fraternity.

Retrieved items:
<item 1>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
After the game, Mintz wrote a well-reported story on how the Braves pulled off the improbable comeback, coming back from down four runs at one point, to win the
game, and deep in the article, wrote the following:

Kevin Pillar's young son, sporting a custom "Money Mike" headband, waddled over to his favorite center fielder for a fist bump. Third-string catcher Chadwick
Tromp strutted by Harris' locker offering a congratulatory "Mike is him!" All while Braves shortstop Orlando Arcia cackled emphatically about Harper's
misfortune, bellowing "ha-ha, atta-boy, Harper!" repeatedly as reporters circled the room.

As Mintz pointed out, he was not the only one to allude to the celebration from Arcia. The Washington Post's Chelsea Janes also wrote about it, saying that one
of Harris' teammates was yelling, "Atta boy Harper."
</item 1>

<item 2>
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
</item 2>

<item 3>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It is also the team’s magnum opus.

Set in a tropical world of zombie pirates, religious fanatics, and talking ships, The Cursed Crew sees you traveling freely across an archipelago as you revive
your undead crew, deploy them on dioramic worlds constituting some of the finest level design in video games to date, and dispatching enemies with a mixture of
tactical stealth and supernatural abilities. Taken at a distance, this mixture is undeniably niche. But seen up close, The Cursed Crew is as potent a creation
as we’ve seen since 2016’s Shadow Tactics: Blades of the Shogun or 2020’s Desperados 3, which respectively established and cemented the studio’s brilliant
design chops. I’ll miss Mimimi — but I can’t imagine a better farewell. —M. Mahardy

17.
</item 3>

<item 4>
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
</item 4>

<item 5>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
This is a story, broken into pieces, we learn. One that ended badly before. One that will likely end badly again. Will you still put it together?

Alan Wake 2 is full of words but few are in order. It is set in places of confused geography, full of people who don’t quite belong, in rooms that might be
different each time you enter them. Alan Wake 2 is wrong, and you must make it right, if you can.

Remedy Entertainment’s assured masterpiece is less about stories and more about dreams — the way they can slip from frightening to absurd at a moment’s notice,
about how we can get lost in them, learning from our experiences or succumbing to insecurities. How video games can mimic their shape, or lack thereof.

It is about repetition, and the way we find meaning in stories and people and places by returning to them over and over, wondering if we changed or they did.
</item 5>

<item 6>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Almost every champion has felt viable throughout the year — no small feat for a game with over 140 playable characters — and it’s led to great fun and variety
on the solo queue ladder and in professional play, where an exciting Worlds just wrapped up.

But there’s another reason League had an outstanding 2023: Arena, a new game mode introduced during the game’s summer event. Arena is a 2v2v2v2 battle mode with
fast-paced chaos, using League’s roster of champions in a more approachable and containable setting (and with less rage-inducing teammates). The mode was
removed after the conclusion of the summer event, but is reportedly returning soon. It can’t possibly come soon enough; I know how I’ll be spending a good chunk
of my winter. —Pete Volk

Related Arcane is officially League of Legends canon now

41.
</item 6>

<item 7>
How queerness became calamity in Tears of the Kingdom (Polygon, 2023-10-19T19:00:00+00:00)
While his developers imagine Link as a blank slate for players of all identities, Sarah Stang contends that Link’s gender expression has never been incorporated
into that vision. “Though featuring an androgynous protagonist can be viewed as progressive in a Western context,” she argues, “Link’s design fits in with
Japanese cultural preferences and was therefore not an overly risky choice for Nintendo.”

Breath of the Wild’s gender trouble continued closer to release, as a series of leaked screenshots revealed Link’s encounter with the character Vilia outside of
Gerudo Town, the all-female city in Hyrule’s desert. The screenshots suggested a transphobic punchline about crossdressing, and the game would deliver on the
joke (the quest being a necessary step to completing the game’s true ending). But when the game was released weeks later, players went on to find queer
expression and even joy in the wilds of Hyrule. A crossdressing Link was genuinely embraced among artists, cosplayers, and authors of fan fiction, and queer
expressions were captured in poetry, art and prose.
</item 7>

<item 8>
Fortnite is the latest game to entice players with a portal to the past (TechCrunch, 2023-10-30T19:18:39+00:00)
Nostalgia sells in 2023

Epic’s decision to turn back the clock might seem small to anyone who hasn’t dropped in off the Battle Bus, but it’s an interesting reflection of the state of
gaming right now. Games — and gamers’ tastes — are increasingly following the wider entertainment trend of digging up familiar stories and repackaging them.

Game developers seemed to believe for a time that cutting-edge graphics and hyper-realism were what consumers really wanted, but gamers’ appetites have trended
toward retro visuals, nostalgia and familiar game worlds more and more. That lower bar for visual fidelity also opens the doors for inventive indie games that
don’t rely on big budgets to shine, but the world’s biggest game developers are riding the wave too.

This year’s hit new Zelda game Tears of the Kingdom literally built a new game on top of the old one and remains a strong contender for Game of the Year.
Competition is stiff though, more so now that Nintendo just casually dropped Super Mario Bros.
</item 8>

<item 9>
Pokémon is no longer just a game — it’s a lifestyle (The Verge, 2023-11-23T18:00:00+00:00)
While it’s easy to mark 2016’s Pokémon Go as the key turning point for Pokémon in its appeal to older demographics — and it certainly was significant — the
framework for this success was paved far earlier by experiments in diversifying what Pokémon could be, whether that be through experiences like Garden or the
transformation of the anime to tell a greater array of stories within this ever-expanding universe.

For the first time, external partners were being brought in to expand the world of Pokémon beyond Ash Ketchum. OLM, long-term Pokémon animation studio since
1998, passed the baton to other studios with the production of a spinoff series: Pokémon Origins. While the studio did assist in production and animated the
finale for the four-episode series, the rest of production was a collaborative effort led by Production I.G and Xebec, and proved to be a hit with audiences
craving something new from the world of Pokémon.

Pokémon Concierge. Image: Netflix

This was just the beginning.
</item 9>

<item 10>
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
</item 10>

<item 11>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
I went into Tears of the Kingdom thinking I wouldn’t build much — I’d just follow the story and get through it; I left feeling like a genius engineer, building
all manner of bizarre contraptions (but mostly long bridges) to sail through skies and trundle over mountaintops.

I still catch my breath remembering that first time diving deep into the darkness of the Depths — the amazement I felt upon discovering a whole other world
underneath the one I knew, filled with skeleton horses and gloom-splattered Bokoblins.

And I remember when I realized where Princess Zelda really was… and then, many hours later, learning where she really was.

I never wanted to stop playing Tears of the Kingdom. I did stop, eventually — the year of 2023 in video games has spoiled us all with hearty meals and sweet
desserts — but I never stopped thinking about it. Every now and then, I picked my Switch back up to seek out another Lightroot, or solve another shrine’s
puzzle, until there were none left.
</item 11>

<item 12>
Super Mario Bros. Wonder review: The joy of pure imagination (Engadget, 2023-10-20T14:11:08+00:00)
The friendly little flowers scattered across each level will even shout words of encouragement or funny quips as you run by. But the impressive part is that,
even in a relatively straightforward platformer like this, there’s a sense of community that makes this world feel more lived in than a lot of epic RPGs.

You’ll find a number of familiar baddies along with some new foes across an incredible range of environments and levels. Wonder features six main worlds each
with a distinctive theme. But within those, you’ll still run into reimagined haunted houses, pirate ships, underwater stages and more. One twist for this game
is the addition of badges, which are earned by completing specific levels and some may even be required to find secrets or reveal alternate exits. Some allow
you to jump farther or float, while others make extra coins appear. There are even protective badges, like the one that automatically saves you if you fall into
lava or poison.
</item 12>

<item 13>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
For me, it’s more a necessity than anything else. So it’s not often that a piece of media — or anything, really — leaves me with the feeling that I need to make
something, to spend time in the kitchen reveling in the tedium of chopping and the sizzle of onions frying.

Venba elicited that urge, reminding me that food is not just something to keep me alive, but something to be cherished. Venba is a cooking game that focuses on
an immigrant family that’s moved from India to Canada. Food transcends the story by way of simple cooking minigames as I move through the chapters of main
character Venba’s life — moments that switch between painful and heartwarming. Venba packs as much heart in its one-hour playtime as games 30 times its size.
—N. Carpenter

Related Venba expands the boundaries of the cooking genre

Top 25

25.
</item 13>

<item 14>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
(It’s up.) Figuring out how to get there is another matter.

Even in games that prominently feature it, climbing is often functional, at best — whether it’s Link refusing to acknowledge handholds are a thing or Nathan
Drake snapping to ledges with a level of magnetism only thought possible at CERN.

Jusant is the first and only game I’ve played that gets it — that sees the inherent grace in the sport and portrays it on the screen, not as a means of getting
somewhere, but as the reason to go there in the first place. —Ari Notis

8. Marvel’s Spider-Man 2

Developer: Insomniac Games

Where to play: PlayStation 5

Marvel’s Spider-Man 2 builds on the successes of the first game, both narratively and in its immensely satisfying gameplay. The game’s story takes a familiar
Spider-Man narrative — that of friends turned foes — and adds new twists, building out compelling villains (and demanding ones) in the process.
</item 14>

<item 15>
Save on the winners from the 2023 Game Awards this weekend (Polygon, 2023-12-09T15:00:00+00:00)
This is a fantastic opportunity to supplement your library with some of the biggest releases of 2023, like Diablo 4 and Starfield, but you can also find
discounts on essential titles that you may have missed, like Valiant Hearts, Her Story, or Celeste. Below, we’ve highlighted some of the winners from the 2023
Game Awards, but we also featured a larger list of winners from previous years.

Cocoon is the latest title from a studio founded by Jeppe Carlsen, the person responsible for the unsettling narrative platformers Limbo and Inside. This
psychedelic odyssey is a personal favorite amongst much of the Polygon staff and won Best Indie Game during the 2023 Game Awards.

The hyper-stylized, rhythm-based action game Hi-Fi Rush was nominated for a number of categories in the 2023 Game Awards and ultimately took home the prize for
audio design, and with good reason.
</item 15>

<item 16>
Super Mario Bros. Wonder is a perfect end for the Switch (Polygon, 2023-10-18T13:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

But, if you’re curious about the “Wonder” — if you’re not too precious about video game tradition — you can find within each Wonder stage a Wonder Seed, a
Mario-sized ball of botany that morphs the environment into something indeed quite wonderful. Like a musical number.

To choose wonder is to flood the atmosphere with lava-lamp plasma, send gravity into convulsions, or transform your hero into any variety of creatures — or even
inanimate objects. As my stoned college roommate once postulated while ripping an apple bong: “You always see the scenery, but have you ever wanted to be the
scenery?” Reader, he dreamt so we could live. For a moment, Mario gets to be the background.

The brilliance of Super Mario Bros.
</item 16>

<item 17>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
But while Nocturne could have easily settled and coasted on classic Castlevania vibes, the new show steps its game up considerably by using the franchise’s
iconic lore to challenge your understanding of what it truly means to fight for freedom, equality, and fraternity.

Set in the late 18th century during the thick of the French Revolution, Castlevania: Nocturne picks up hundreds of years after the events of Netflix’s first
Castlevania series as demon slayer Richter Belmont (Edward Bluemel) makes his way to France in search of vengeance and vampires. Like his distant holy
whip-wielding ancestor Trevor, Richter’s innate talents for magic make him perfectly suited for getting into the family business, and being raised by his
formidable warrior mother Julia (Sophie Skelton) instills in him a profound sense of pride in the work they do.

Image: Netflix

Together, Richter’s raw potential and Julia’s tutelage make him uniquely powerful even from a young age.
</item 17>

<item 18>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
This year, we have the Avengers of rhythm games, Theatrhythm Final Bar Line, and Paranormasight: The Seven Mysteries of Honjo, an excellent riff on the visual
novel penned by a beloved storyteller — whose best series has never appeared in the U.S.

What should you know about Paranormasight before you play? Well, ideally nothing. Why else would I be eating up my word count?

But if you insist: It’s a mystery — and a horror mystery at that. You travel to 1980s Japan, specifically the Tokyo neighborhood of Honjo, located not so far
from the modern Tokyo Skytree. It’s hard to imagine that modern landmark ever towering alongside these streets, which are filled with shadows and lethal curses.

If you have even a passing interest in urban legends, spooky folklore, cults, and deadly rituals, or you’ve enjoyed series like Zero Escape and Danganronpa,
Paranormasight is an easy recommendation.
</item 18>

<item 19>
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
</item 19>

<item 20>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
But in all the bloodshed, Nocturne still finds the time to make its characters (and by extension, you) think about what it means to slaughter technically
innocent people for a genuinely noble cause. That beat isn’t entirely original, and Nocturne’s mindful about leaning on it too hard as it begins to weave
Annette and Edouard’s stories together with Richter and Maria’s in France. Still, it creates a resonance between the show’s action and the concepts it’s
tackling that makes the season as a whole work beautifully.

Image: Netflix

Occasionally, there are moments where the fluid dreaminess of Nocturne’s overall art direction butts up against the show’s application of 3D animation to make
certain pivotal moments pop. It can be jarring when traditionally animated characters who were just popping in and out of view using tricks of the eye suddenly
turn to the camera with deeper faces accented by flat shadows that weren’t there a second before.
</item 20>

<item 21>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
And then I’d just wander, collecting ingredients, talking to Great Fairies, imagining the next adventure.

Tears of the Kingdom feels like someone holding my hands very close as they lean in to whisper, with eyes twinkling, “Can I tell you something?” The ride was
wild; I laughed, I cried. And I can’t wait for the next time, when it’s completely different. —M. Myers
</item 21>

<item 22>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
The only difference was that Janes did not name the player.

The reporting of Arcia's comments proved to be divisive across the baseball community. Among the most prominent to call out Mintz was Rizzo. The "High Heat"
co-host said she was irritated by "bloggers or podcasters" going into the clubhouse and "ruining it for the rest of us" when "some jackoff comes in at the end
of the season that gets a credential, God only knows why." She called the clubhouse "a sacred place" and defended the players who were upset by the reporting of
Arcia's comments, before she called Mintz out by name.

MORE: Every time Nick Castellanos has interrupted a somber moment with a home run

"First of all, this guy, Jake Mintz, that's not even a reporter," Rizzo said. "That's taking away from true reporters and true journalists.
</item 22>

<item 23>
Super Mario Bros. Wonder is a perfect end for the Switch (Polygon, 2023-10-18T13:00:00+00:00)
Galactic travel! 2D Mario, conversely, defaulted to a vessel for nostalgia.

I can say with confidence that Wonder shares those early games’ inventiveness, and their disregard for what should and shouldn’t be in a Mario game. When
playing Wonder, I’ve imagined how it came to be: In my fantasy, since the ’90s, every time a Nintendo developer had an idea that “didn’t fit the tone” or
“wasn’t Mario enough,” they wrote it into a notebook. And eventually, with no space left, someone had the gall to say, “Let’s turn this stuffed notebook of
chaos into a game. Because why shouldn’t Mario be an elephant? Why shouldn’t Wigglers wear roller skates? And why shouldn’t Luigi skydive?”

Wonder’s creators have so many new ideas to share that their creativity is unapologetically wasteful, which I say as a compliment. Many gimmicks in Super Mario
Bros.
</item 23>

<item 24>
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
</item 24>

<item 25>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
Part of / The best entertainment of 2023

Netflix’s first Castlevania animated series excelled by balancing its quiet, beautifully performed moments of human drama with explosive and artful action set
pieces dazzling enough to evoke the feeling of playing Konami’s classic side-scroller for the first time. Despite the first show’s strengths, its regular issues
with pacing and effectively cultivating characters from season to season made the idea of Netflix keeping the vampiric daddy issues ball rolling with
Castlevania: Nocturne — a new spinoff series from executive producer Clive Bradley and co-directors Sam Deats and Adam Deats — a dubious one.

Nocturne both looks and feels a lot like its predecessor in terms of its focus on a tortured scion waging war against machiavellian, bloodsucking ghouls who
fancy themselves undead gods who walk among mortal livestock.
</item 25>

<item 26>
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
</item 26>

<item 27>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
And if you just enjoy a good yarn and have access to basically any screen and $15, then you’re a perfect mark too. It runs as well on console and PC as it does
on iOS and Android, so don’t fret about where you play, just do so and soon! Before Square Enix stops investing in all these oddities. —Chris Plante

22. Marvel’s Midnight Suns

Developer: Firaxis Games

Where to play: Nintendo Switch, PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

[Ed. note: Marvel’s Midnight Suns was released in 2022, but it just barely missed the cutoff for our best video games of 2022 list, so it’s eligible for our
2023 awards.]

I know what you’re thinking: Another licensed Marvel game? Come on, right? But hear me out. I played Marvel’s Avengers, too, and this isn’t that.
</item 27>

<item 28>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
By Chad Jennings, C. Trent Rosecrans and Stephen J. Nesbitt

In one American League Wild Card Series, the whole thing turned on a play designed around crowd noise. It was too loud at Target Field, Twins shortstop Carlos
Correa realized, for the third-base coach to warn Blue Jays baserunners about a developing pickoff play. The Twins used the noise to their advantage, picked off
Vladimir Guerrero Jr. and secured a two-game sweep.

Advertisement

In the other AL Wild Card Series, empty seats were everywhere. The Rays are an excellent team that thrives on their underdog status — low payroll, injured
players, they always find a way — but playing in front of two of the smallest postseason crowds in the past 100 years, the Rays were swept by a Rangers team
that had nine fewer wins in the regular season. The indifference was deafening.

Home-field advantage, it turns out, just isn’t the same from one ballpark to the next.
</item 28>

<item 29>
Super Mario Bros. Wonder review: The joy of pure imagination (Engadget, 2023-10-20T14:11:08+00:00)
You’ll never be bored, even if you want a change from standard platform jumping.

I also have to say that this might be the best Mario and friends have ever looked, and that includes recent 3D titles like Odyssey. There is just so much detail
packed into every level and character. From the determined face Mario makes when he runs around, to the little flames that sprout from your feet every time you
jump.

Wonder’s map is also surprisingly engaging. In the beginning, I wasn’t sure if the ability to freely roam around added all that much. But as I progressed, I
found a number of secret nooks and crannies that wouldn’t have fit if the game was locked to a handful of paths. And with Wonder Flowers giving Nintendo the
freedom to mix and match mechanics and enemies, there’s just so much to see.

Don't let Wiggler beat you. Screenshot by Sam Rutherford/Engadget

Nintendo even managed to address a gripe I had about the pacing of previous 2D Marios.
</item 29>

<item 30>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
(If you can get past the Frankenstein-ification of the boroughs, that is. RIP Chelsea.)

There are hints of greater thematic aspirations within Spider-Man 2. Questions about mass incarceration, recidivism, and reintegration into society for formerly
incarcerated people; worries about the surveillance state and what we give up when we engage with modern technology. These things appear, however fleetingly, in
Spider-Man 2, but there isn’t enough time to elaborate on any single one in the unrelenting march of set-piece after glorious set-piece.

But I’m glad they’re there, anyway. Because if we’re going to retell a story, we ought to retell it anew. Canons were made to be broken, and Spider-Man 2 swings
in that direction. With a sequel teed up by the game’s final act, I’m eager to see where Insomniac goes with that momentum. Next stop: the Bronx.

Marvel’s Spider-Man 2 will be released Oct.
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
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
As we donned Samus’ suit and explored strange planets, aggressive alien lifeforms could now get right in our faces, forcing us to dodge, strafe, and roll (in
morph ball form, naturally) using all three dimensions. No longer would we sit back and watch as Samus dipped her toe into a pool of lava; in first-person, as
molten fire spread over our visor, we’d really feel the pressure to find that Varia Suit upgrade. And perhaps most importantly, from behind Samus’ visor, we
gained the ability to scan our enemies and environment, collecting and translating logs from the long-dead Chozo aliens who once inhabited these now-hostile
places.

The world of Prime is harsh and unrelenting. (Save points will, at times, be quite far from one another.) But it’s worth buckling down and pushing through the
pain points to discover this world’s secrets. —M. Myers

18.
</item 32>

<item 33>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
It had been reported Monday night when the Braves beat the Phillies in Game 2 of the NLDS, ending on a critical base-running play involving Bryce Harper, that
Atlanta shortstop Orlando Arcia taunted about the Phillies' star in the clubhouse after the game.

That report wound up generating plenty of controversy.

Fox Sports' Jake Mintz, one of the two brothers that runs the popular Cespedes Family BBQ account, reported that Arcia had been celebrating after Atlanta's 5-4
win by saying "Atta boy Harper" in the Braves' clubhouse, jeering at the fact that Harper was doubled off on the basepaths after an outstanding catch by Michael
Harris in center field and an impressive relay to get Harper at first to end the game.

The comments wound up coming back to bite the Braves two days later, with Harper homering twice — staring down Arcia after each — as he powered the Phillies to
the 10-2 Game 3 win.
</item 33>

<item 34>
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
</item 34>

<item 35>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 35>

<item 36>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
The game’s narrative remains intact, no matter what. There never occurs a feeling of the devs trying too hard to make it seem like her missions are stretching
out of their boundaries.

Although they are designed well this time but one has to commend Insomniac’s team to keep a limit on not making MJ’s gameplay moments seem forced or going
beyond a certain point. They serve their purpose without affecting the main plot of Marvel’s Spider-Man 2 or overlapping with Peter or Miles’ stories. In a way,
each character has been given a good amount of time to shine. At times, it may feel like MJ’s playable sequences should have been extended. But it’s nice to see
that the devs decided to not do that; sometimes less is more and fine.

Despite Spider-Man 2 being more of a Peter’s story (in its entirety), MJ’s role has been given equal significance too. There’s a visible growth in her arc
especially considering her profession as a journalist.
</item 36>

<item 37>
The best new TV series on every streaming service (Polygon, 2023-10-25T16:09:11+00:00)
Our latest update added Castlevania: Nocturne, Gen V, Hijack, and The Continental.

The best new shows on Netflix

Castlevania: Nocturne

Creator: Clive Bradley

Cast: Edward Bluemel, Thuso Mbedu, Pixie Davies

To say that the follow-up to Netflix’s Castlevania had a lot to live up to would be an understatement. Based on Konami’s acclaimed action-adventure game series
and produced by Texas-based Powerhouse Animation Studios, the original 32-episode adult animated series was an unmistakable success for the streamer when it
premiered in 2017. Chronicling the adventures of Trevor Belmont, the last surviving descendant of a long line of monster hunters, and Sypha, a powerful magic
wielder hailing from a reclusive tribe of magicians, the series followed the pair as they joined forces with Alucard, half-human son of Dracula, to slay his
vengeful father and save humanity from the threat of extinction.
</item 37>

<item 38>
Super Mario Bros. Wonder review: The joy of pure imagination (Engadget, 2023-10-20T14:11:08+00:00)
Mario has been stomping Goombas, smashing blocks and running from left to right for damn near 40 years. How is Nintendo supposed to keep its classic side
scroller feeling fresh after all this time? First, you shift the location to a brand new kingdom. Next, you introduce a bunch of new enemies and power-ups. But
what really pushes things over the top is the introduction of a new mechanic that turns everything you think you know about Mario games on its head. What you
get is Super Mario Bros. Wonder, a game that dazzles and delights while living up to its name in a variety of ways.

Warning: This review includes some light spoilers, including the number of worlds and descriptions of some hidden levels.

The game starts simply enough with Mario and crew (Luigi, Peach, Toad, Yoshi, Nabbit and, for the first time, Daisy) visiting Prince Florian of the Flower
Kingdom.
</item 38>

<item 39>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
Win and it goes away. Period."

The incident elicited a response from the Baseball Writers Association of American (BBWAA), which defended Mintz, a BBWAA member.

"For the record, Jake is a BBWAA member in good standing and has earned the right to be credentialed through his hard work and quality baseball coverage," the
BBWAA wrote in the statement. "To assert otherwise, in vulgar terms, is both unprofessional and unacceptable. The BBWAA is deeply troubled that the league's own
network would permit the disparaging of one of our members in this fashion. Scrutinizing our work is part of the territory but comments such as these should
have no place on MLBN."

Following the backlash to her comments, Rizzo apologized on "High Heat" to Mintz and the BBWAA. She said her comments were "completely inappropriate, it was
inaccurate and it was beyond unprofessional.
</item 39>

<item 40>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
I played Marvel’s Avengers, too, and this isn’t that. It might seem like it’s going to be at first, because Midnight Suns makes the grave error of introducing
Iron Man and Doctor Strange as its tutorial characters, and these two might just be the most irritating characters in the entire video game. (I have beaten the
game, so I am allowed to make this call.) You must press on and give Midnight Suns time to win you over. Because it has so, so much more to offer than it may
appear in its first few hours.

Picture the romance and humor of Fire Emblem: Three Houses, combined with the high-stakes tactical battles of XCOM 2 — that’s what Midnight Suns becomes in its
mid-game and endgame. It’s a card-based strategy game, and each hero has their own customizable deck. I started off favoring Captain Marvel, Magik, and Blade,
simply because their moves and hilarious dialogue kept me entertained, but I soon realized that every single character has something exciting or unexpected to
bring to the battlefront.
</item 40>

<item 41>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
“A small subset of teams are good to go for the next 10 years, but others have these peaks and valleys, and we were somewhere in between,” Krankel said. “We
weren't in danger of anything going sideways. But we were at a spot where we're like, it would be cool to tether to somebody who has a similar vision, and
somebody that we could work with that would like, de-risk us.”

Netflix

This is the short-term benefit of being bought by a larger company, but there are downsides to relinquishing independence. Having a corporate overseer can
result in rigid production timelines, hindering a studio’s ability to pivot, and despite all of the promises otherwise, developers may be forced to adhere to a
specific tone, vibe or game-development structure. Owned studios are held accountable by people outside of the actual development of a game, and the bigger the
company, the further away its bosses are from the creative process.

The most extreme negative outcomes for an acquired indie studio are, of course, layoffs and closures.
</item 41>

<item 42>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
The Max Fried-Jorge Soler game in 2021 happened in Houston. The iconic moment at Truist Park could be Eddie Rosario’s dazzling catch at the wall in Game 4 of
the 2021 World Series, but the most memorable — even though it didn’t lead to a championship — might be rookie Ronald Acuna Jr. hitting a grand slam off Walker
Buhler in Game 3 of the 2018 NLDS, which happened to be the first postseason game ever played at the ballpark.

Our take: Obviously, there are issues bigger than baseball to deal with here. The war chant and Tomahawk Chop are offensive and problematic, but we’re not going
to resolve that issue in the next four weeks. In fact, it’s all about to come front and center — again — as the best team in baseball opens its postseason with
a roster that would be plenty intimidating in a Little League park with metal bleachers.
</item 42>

<item 43>
Microsoft Activision: What does deal mean for gamers? (BBC News - Technology, 2023-10-13T23:01:24+00:00)
Meanwhile Sophie, who was named Ubisoft's content creator of the year, told the BBC she "just wants the games to win".

"We want everyone to be able to play video games that they want to play, we want them to be accessible for everyone," she said.

"Obviously, we're going to have our opinions, but if the end result is good games, that's fine with us."

This Instagram post cannot be displayed in your browser. Please enable Javascript or try a different browser. View original content on Instagram The BBC is not
responsible for the content of external sites. Skip instagram post by cadaea Allow Instagram content? This article contains content provided by Instagram. We
ask for your permission before anything is loaded, as they may be using cookies and other technologies. You may want to read Meta’s Instagram cookie policy,
external and privacy policy, external before accepting. To view this content choose ‘accept and continue’. Accept and continue The BBC is not responsible for
the content of external sites.
</item 43>

<item 44>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
For me, Project ECCO found that balance.

A story of time travel and cosmic horror, Project ECCO is easy to get into and relatively quick to play (three to 10 hours per game). Between the hundreds of
narrative prompts, unlocking various time travel “devices” (coins, dice, tarot cards, etc.), and the absolutely stunning layout, this is one of my favorite
TTRPGs, period.

Played in the pages of a yearly planner, Project ECCO hits every beat of a good time-travel story: time loops and timeline divergences, a shady and controlling
Time Travel Agency, and an unknowable cosmic Entity consuming spacetime. As you play, you (literally) burn through the days of your planner, creating an
artifact, marked and changed by the telling of your agent’s story.

For an example of gameplay and a masterfully produced actual play, listen to the Project ECCO miniseason of My First Dungeon with game designer Elliot Davis.
</item 44>

<item 45>
Super Mario Bros. Wonder review: The joy of pure imagination (Engadget, 2023-10-20T14:11:08+00:00)
But then Bowser barges in and steals a Wonder Flower, which somehow allows him to merge with Prince Florian’s castle to create a floating mecha fortress of
doom. From there, your job is to collect Wonder Seeds (this game’s version of stars and shines) as you help the Florians restore order to their troubled land.

Screenshot by Sam Rutherford/Engadget

Right away, this has some subtle but important implications for the rest of the game. The first is that you immediately Wonder Flowers have wild and
unpredictable effects. The second is that, because you’re not simply recusing a princess trapped in a castle, the whole Flower Kingdom is working with you to
take down Bowser. Sometimes that means you might be rescuing some trapped miners. But other times the Florians are the ones helping Mario (who also has a new
voice actor for the first time) by rebuilding a bridge or donating a Wonder Seed at the beginning of a new world.
</item 45>

<item 46>
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
</item 46>

<item 47>
The best apps and tools for managing your money online (The Verge, 2023-11-12T13:00:00+00:00)
I want to know what you’re into right now as well! Email installer@theverge.com with your recommendations for anything and everything, and we’ll feature some of
our favorites here every week.

“There’s a website called RetroAchievements that adds trophies to old-school games. It’s a great excuse to go back and play games from your childhood or games
you never got around to playing.” – Nick

“Fluttermind, the Moonring dev, hasn’t ported the game to Mac yet, but I did discover some of their other games, and I’ve started to play Spellrazor. It’s a
very interesting haunted arcade game.” – Drake

“Just finished The Kids of Rutherford County, the new podcast from the Serial folks, and it was great. Reading Adam Grant’s new book, Hidden Potential. Not done
yet, but it’s excellent so far, especially for someone like me who struggles with imposter syndrome.” – Nick

“Castro, my favorite podcast app.
</item 47>

<item 48>
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
</item 48>

<item 49>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The idea is that, since you’re never that far from flat ground, completing your route is more a game of mind over matter. You don’t “finish” a problem. You
“solve” it. Like a puzzle.

When it comes to rock climbing, Jusant just gets it in a way few other games — if any — have.

To be clear, Don’t Nod’s Jusant isn’t technically about bouldering. (A climbing game where you’re never more than 12 feet off the ground would have precisely
zero stakes.) But it’s an apt conduit for the spirit of bouldering, in that every route you traverse is a problem to solve. By alternating triggers to dictate
which hand goes on which handhold, you navigate these problems. Environmental hazards and a pesky stamina wheel make things progressively trickier. You’re
always clear on your heading. (It’s up.) Figuring out how to get there is another matter.
</item 49>

<item 50>
Dropout’s dropping CollegeHumor on its 5th anniversary — here’s why and what the future holds (Polygon, 2023-09-26T17:35:00+00:00)
Burrow’s End is a really fun season of Dimension 20. And I say fun because Aabria’s taking us to this wonderful Secret of NIMH, Watership Down — there’s a
certain subgenre. And we find in Dimension 20 that we have our most success when we get specific. It’s not just fantasy, it’s Game of Thrones meets Candyland.
It’s not just a mystery, it’s noir, or it’s Agatha Christie — which is actually two of the Side Quest seasons we’ve done, one of which Sam was a player in. It’s
about getting specific with the comparisons that you’re reaching out to and where you’re drawing inspiration from.

“Cute little critters” is a quite broad, fantastical storytelling genre — within that there’s a subgenre of perhaps the darkest and most grim stories ever told.
</item 50>

<item 51>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
It’s a war between undead ghouls like Drolta Tzuentes (Elarica Johnson) manipulating mortals from the shadows, selfish humans like a power-hungry Abbot
(​​Richard Dormer), and agitators like summoner Maria (Pixie) who want to dismantle the entire oppressive system of power that makes France a formidable global
presence.

Image: Netflix

Narratively, Nocturne’s focus on another brooding Belmont and its depiction of late-18th-century France as a country overrun by literally vampiric aristocrats
drunk on the promise of a new messiah make the new series feel true to the essence of Castlevania. What makes Nocturne’s deep dive into this alternate history a
worthwhile watch, though, is the way it presents France not just as a crumbling monarchy in the throes of a heroic social uprising but also as a colonial nation
whose pivot to democracy at home can’t be divorced from its history of committing unfathomable atrocities abroad.
</item 51>

<item 52>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“The Taylor Swift and Travis Kelce news has been a pop cultural moment we’ve leaned into in real time, as it’s an intersection of sport and entertainment, and
we’ve seen an incredible amount of positivity around the sport.”

“The vast majority of our content has remained focused on the game, our players and variety of other initiatives, including our Toy Story Funday Football
alt-cast, the international games and more,” they continued.
</item 52>

<item 53>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Or, to give a more mass media example, like Star Wars: Visions, which gives creative leeway to a variety of animators and storytellers to play with the very
basics of what makes Star Wars Star Wars.

Spider-Man 2 doesn’t go quite this far, but where it does go is still satisfyingly slant. Wrinkles to critical characters and plot points will keep even
seasoned Spider-fans on their toes. Moreover, it plays like a dream, with smart additions to combat that enhance the power fantasy of being Spider-Man. Each
Spider-Man now has a full arsenal of special moves and gadgets that complement one another, making even the most basic combat encounter feel like a
choreographed ballet of kicks, thwips, and zaps. The game is beautifully rendered, taking full advantage of the newest PlayStation, with such detailed textures
on each unlockable suit that you’d swear they were real, and ray-traced window after ray-traced window in a loving rendition of New York.
</item 53>

<item 54>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
"That's taking away from true reporters and true journalists. Where were you on the random Tuesday in April in Cincinnati when this team was playing in
Cincinnati? Where were you on a rain delay, and you have to sit in the press box for all of this time? It's ridiculous."

"And then some jackoff comes in at the end of the season that gets a credential, God only knows why. And the clubhouse is a sacred space." Alanna Rizzo went off
on Jake Mintz's clubhouse reporting on MLBN's "High Heat" Thursday. pic.twitter.com/6dooqeLPnh — Awful Announcing (@awfulannouncing) October 12, 2023

Rizzo's comments immediately drew heat from the baseball journalism world, including from Janes, who posted a thread defending Mintz. Janes reported that
cameras and recorders were recording, and that Arcia yelled the comments while they were recording and with more than a dozen reporters around.
</item 54>

<item 55>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
As strong and different as the season starts, by its last two episodes, Nocturne does start to feel a bit like it’s rehashing moments from the first show — if
not in exact form, then in terms of how neatly it brings its plotlines to a close. None of that’s enough to keep the show from hitting all the right notes,
however, and by the time you hit the finale, there’s a solid chance you’ll already be hoping for more.
</item 55>

<item 56>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Everything is filtered through the titular main character Tchia’s eyes, eyes with a special power that allows her to transform into any animals or objects in
her environment. Birds, dolphins, a camera, or rocks… It’s all an option for Tchia.

The game, while clearly inspired by The Legend of Zelda: Breath of the Wild, ends up standing on its own because of the innovative shapeshifting mechanics.
Tchia isn’t as technically polished as a Nintendo title with hundreds of developers; Awaceb has a team of roughly a dozen. Still, it’s hard to innovate in such
a ubiquitous genre, yet Awaceb has managed to do just that with Tchia, making it one of the best games so far this year. —Nicole Carpenter

47. Blasphemous 2

Developer: The Game Kitchen

Where to play: Nintendo Switch, PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

Long gone are the days when the Metroidvania genre languished untouched for years on end.
</item 56>

<item 57>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
So an age-old hero in a green tunic has to step in. We all know the beats — how could you possibly make that into anything new?

Tears of the Kingdom definitely didn’t seem like it was poised to make that happen: a sequel to one of the most popular Zelda games ever made, originally
conceived as DLC, and built on the same map. Yet, it is something else entirely. It’s not just that Breath of the Wild was a rough draft for Tears of the
Kingdom — it’s that the entire Zelda series was a collection of stepping stones that led in winding, influential pathways to this wacky, wonderful, and
thoroughly new world of Hyrule.

I mean, Ultrahand alone. Just absolutely slathering pieces of wood in what amounts to magical Gorilla Glue and watching in wonder as the game’s physics engine
roars to life in response.
</item 57>

<item 58>
The best new TV series on every streaming service (Polygon, 2023-10-25T16:09:11+00:00)
Castlevania: Nocturne picks up three centuries after the original, and while Trevor and Sypha may be dead, their legacy lives on in the series’ new protagonist,
Richter Belmont. Set amid the French Revolution, Nocturne follows Richter as he is enlisted by a young sorceress named Annette to fight against a conspiracy of
aristocrats who have allied themselves with a powerful enemy known as the Vampire Messiah. Haunted by the death of his mother at the hands of a mysterious
vampire named Olrox, Richter must call upon every ounce of his courage to face this new darkness and assume his birthright as a Belmont. Castlevania: Nocturne
is a terrific installment in the franchise that meets and exceeds the original Castlevania at its best. From its brilliantly realized character designs and
exhilarating action sequences to its beautiful score and nuanced take on real-life revolutions, the series is both an excellent continuation and fantastic entry
point for new audiences.
</item 58>

<item 59>
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
</item 59>

<item 60>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Galaxy-spanning faction quests give you plenty to do, but it’s the side quests that really make you feel like the captain in something Star Trek adjacent:
generation ships and superhero hideouts and colonies of clones. There’s good sci-fi here for those willing to make the journey.

It’s true that the game oversold the idea of a galaxy with over 1,000 planets, a number that pales in comparison to the game’s most obvious competitor, No Man’s
Sky. It’s a shame this became so much of the focus thanks to Bethesda’s marketing, because that same focus obscured the absolutely fascinating new game plus
mode at the game’s heart. Consider this: No Man’s Sky is the game that realizes the universe’s infinite number of planets, while Starfield realizes the
universe’s infinite number of possibilities. —Clayton Ashley

33.
</item 60>

<item 61>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It’s very rare for a Soulslike to ever feel like anything more than a knockoff — even when they’re decent fun, like The Surge. But the best compliment I can
give Lies of P is that it feels like the genuine article, a FromSoftware game developed in an alternate dimension and somehow released in this one by mistake.
But it wasn’t a mistake or luck that made Lies of P, and it wasn’t FromSoftware, either; it was a talented group of developers at Neowiz Games and Round8 Studio
that took a tired genre, paired it with a bizarre IP, and knocked it out of the park. —Ryan Gilliam

Related Lies of P carves a singular space out of the Soulsborne genre

48. Tchia

Developer: Awaceb

Where to play: PlayStation 4, PlayStation 5, and Windows PC

Tchia is an open-world adventure game set in a fictional version of island nation New Caledonia — inspired by Awaceb’s co-founder’s childhood in the country.
</item 61>

<item 62>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
But despite all of Julia’s skill with her family’s deadly heirloom, Castlevania: Nocturne’s world is one teeming with vampires even centuries after Dracula has
seemingly been destroyed, and the series opens at a devastating moment in Richter’s past that leaves him orphaned and unable to access his magic.

Though Nocturne takes place in the same continuity as the previous Castlevania show, it also uses major pieces of lore from Konami’s Castlevania: Rondo of Blood
and Symphony of the Night games to tell a riveting story about the inherent hypocrisies baked into the historic record of colonial nations embracing democratic
revolution.

Here, the French Revolution isn’t just a transformative period of social unrest between moneyed elites and impoverished peasants fed up with their subjugation.
</item 62>

<item 63>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Revisiting the NBA caricature back in time, the New York Knicks became the first all-Black team in 1979. But their fans made their disapproval very apparent.
“It wasn’t so much New York attitudes, but fans around the league let you know how they felt about you as an opponent and as a Black man as well,” Knicks star
Earl Monroe told Newsday.

USA Today via Reuters May 15, 2021; San Antonio, Texas, USA; Phoenix Suns forward Torrey Craig (12) and San Antonio Spurs forward Keldon Johnson (3) fight for
loose ball as Suns guards Devin Booker (1) and Chris Paul (3) look on in the third quarterat AT&T Center. Mandatory Credit: Scott Wachter-USA TODAY Sports

We could list out all the scandals, fights, and controversies here, but you get the idea.
</item 63>

<item 64>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
wasn’t consigned to the newcomers, though. Larian Studios, fresh off two years of early-access development and riding the reputation it had garnered from
Divinity: Original Sin 2, saw fit to release a role-playing game in which you can kill off nearly every main character the moment you meet them. Remedy
Entertainment — let’s be honest, this group has always been strange — made a sequel that’s equal parts horrifying, hilarious, fun, and fabulous. And Nintendo?
Well, Nintendo had another banner year. That’s no surprise. The real surprise? It finally let go, and let players toy with the digital molecules of its most
revered series. More on this below.

As the year comes to a close, it’s intoxicating to see developers of all sizes, in every genre, with every tier of budget, mining the depths of interactive
design, branching this way and that as they follow their respective veins of gold. They’re nowhere near the bottom of that particular expanse, of course — and
that’s a heartening thought.
</item 64>

<item 65>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
Here's what you need to know about the controversy.

Bryce Harper Atta Boy controversy, explained

It all begins with the play. In Game 2, Harper led off the top of the ninth in a one-run game with a walk. J.T. Realmuto flew out to center for the first out,
but on the next play, Nick Castellanos lifted a deep fly into right-center. It appeared it would be extra bases, and if Harper took off, he'd be able to score
the tying run on the play.

The problem for the Phillies was that Harris made a leaping grab up against the wall to rob Castellanos of the hit. Harper had well since rounded second on his
way to third, but the throw from Harris missed both cut-off men by second. Austin Riley, however, ran in and scooped the ball and made an impressive throw to
Matt Olson to nab Harper and end the game.
</item 65>

<item 66>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The best games this year told their stories in interesting ways, be it ignoring your superpowers to bike through your old neighborhood in Queens, a musical
number, or fractured memories hidden in a puddle. But Baldur’s Gate 3 takes experimental storytelling to another level. Every choice and every new direction is
a simple success, fail, or crit away. One roll of the dice after another can take you all the way from a crashed Nautiloid to the city of Baldur’s Gate itself.

In a year filled with beautiful stories and powerful moments, it’s the spinning whir of Baldur’s Gate 3’s d20 that defined video game storytelling in 2023. —RG

1. The Legend of Zelda: Tears of the Kingdom

Developer: Nintendo EPD

Where to play: Nintendo Switch

Princess Zelda is missing. Ganondorf has returned, somehow, and he’s really hot. So an age-old hero in a green tunic has to step in.
</item 66>

<item 67>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
There have been many, many games in the franchise, but none have reached the heights of Armored Core 6: Fires of Rubicon. The levels have never had such ruined
beauty, the enemies have never been as satisfying to fight, and the characters have never been more endearing. An ingenious new game plus mode packs the game’s
story with surprises while you continue to build out your arsenal.

Perhaps the decades-long wait for a new Armored Core game is part of the reason, because the lessons FromSoftware learned creating and popularizing Dark Souls
are evident here. Bosses now put you through dastardly skill checks in classic Soulsian fashion. Many of them tower over the landscape, making you, the pilot of
a giant robot, feel small. These set-piece fights are some of the most thrilling moments you could play in a game this year. They certainly have some of the
most memorable lines.

But Armored Core 6 doesn’t just get badass one-liners stuck in your head; its gameplay lingers, too.
</item 67>

<item 68>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
Nocturne reworks a number of Castlevania’s classic supporting characters like Maria — who fights using reimagined versions of the Four Sacred Beasts — and
empathic singer Edouard (Sydney James Harcourt) to brilliant effect. But out of all the show’s old but new faces, ferrokinetic Vodoun practitioner Annette
(Thuso Mbedu) stands out because of how strongly the show introduces her as a woman whose Haitian identity is what makes her able to understand the situation
over in France with an exacting kind of clarity.

It’s through Annette and the show’s exploration of Afro Caribbean syncretism that Nocturne expands the boundaries of Castlevania’s magical mythology, which is
touched on heavily throughout the season’s eight episodes. Small nods to the video games and gnarly beheadings abound, particularly in the Powerhouse
Animation-produced show’s bigger hyperkinetic brawls between its heroes and hordes of horrific creatures.
</item 68>

<item 69>
Super Mario Bros. Wonder review: The joy of pure imagination (Engadget, 2023-10-20T14:11:08+00:00)
And I’m slightly disappointed the adorable penguins from the movie didn’t make an appearance somewhere in Wonder’s ice levels.

Have you ever seen a Boo this big? Screenshot by Sam Rutherford/Engadget

But on the whole, Super Mario Bros. Wonder is a marvelous game. Its art and animations have a level of polish that’s practically unmatched by rival titles, and
the addition of Wonder Seeds subverts your expectations and keeps this 40-year franchise feeling new and innovative. In a lot of ways, Wonder reminds me of my
favorite 2D Mario game: Super Mario World. While I wish it didn’t take more than 10 years for Nintendo to create a proper sequel to its last mainline
platformer, if that’s what it takes to get a game like this, I’m happy to sit and wonder what’s next for Mario.
</item 69>

<item 70>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
If you want curated lists of our favorite media, check out What to Play and What to Watch .

Retellings are compelling because they allow writers to toy with readers’ preconceived notions of how the story “should go,” both in terms of genre conventions
and in terms of plot. We know Little Red Riding Hood will go to her grandmother’s house, where she will find that her grandmother has been eaten by a wolf, just
as we know Spider-Man will be bitten by a radioactive spider. The fun, both for the author and the reader, comes in deviating from those norms — in finding new
ways to tell old tales.

With Marvel’s Spider-Man 2, Insomniac Games attempts something similar.

Mainstream Spider-Man stories are increasingly Spider-Men stories, and Spider-Man 2, the third Spidey game from Insomniac, is no different.
</item 70>

<item 71>
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
</item 71>

<item 72>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
For the first year in recent memory, scale did not necessitate tradition, and scope did not preclude getting weird. In 2023, nothing was sacred in video games,
and so they felt more vibrant than ever.

Sure, some of the more “focused” games threw us for a pleasant loop: Dredge begins as a lonely fishing sim before transforming into something otherworldly, and
Humanity morphed from a pensive art project into an all-out war. Dave the Diver, similarly, is not so much about being a diver as it is about running a sushi
restaurant, or hunting for alien artifacts, or conversing with said aliens, or — you get the point. Whether you booted up your Steam Deck for a cross-country
flight or hid your Switch off screen during that boring Zoom meeting, the game you returned to was rarely the one you left behind.

This amorphousness (I’m begging our copy editor to let this one slide, because what other “word” could adequately summarize the video games of 2023?)
</item 72>

<item 73>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The book itself has a stunning and distinct art style that immediately transports you into the doom metal fantasy of the setting; but that’s only a small part
of its allure.

Unlike many TTRPGs that one merely has aspirations of playing one day, Mörk Borg does everything in its power to make the game easy to run. Although it’s a
short read, no line is wasted. Even item descriptions are dripping with hints at nefarious deeds one must have done to acquire them. The optional classes are an
absolute delight for those of us that savor the idea of playing the unsavory. From the esoteric introverts that choose to make their hovel far from prying eyes
to the unfortunate discarded-at-birth cutthroats who’ve had to scratch out a living any way they can, there is something for everyone. There is always the
option of creating something more tailor-made if none of the classes are appealing, but I appreciate that Mörk Borg offers beginner-friendly ideas and
archetypes to help you create something that fits right into the gritty setting.
</item 73>

<item 74>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The game is a remarkably inviting take on a genre not exactly known for being accessible. But Hi-Fi Rush does away with hardcore precision in exchange for
gameplay that always subtly nudges you back toward keeping in time. Failing to attack on a drum stroke doesn’t mean failing a fight; you just don’t get a combo
multiplier. And the musical score never gets jarring as a punishment; punchy notes always play in time with the rhythm, even if you hit a button at the wrong
time. The result is a game that evokes the thrill of feeling like you’re acing it and getting into a flow, no matter your skill level. —N. Clark

29. Monster Hunter Now

Developer: Niantic

Where to play: Android and iOS

While Niantic’s post-Pokémon Go track record hasn’t been the most consistent, Monster Hunter Now shows the company at its best, with a refined combat system,
simple matchmaking, great monster variety, and extensive character upgrades.
</item 74>

<item 75>
A field guide to the vampire authors who were probably vampires (Polygon, 2023-11-09T17:12:20+00:00)
Vampire fiction has a long and distinguished pedigree in English. From the moment that Robert Southey introduced the British Isles to the concept of the
coffin-sleeping undead at the turn of the 19th century, the authors of vampiric stories have mirrored the changing face of the society that reads them.

We're spending a week diving deep into the stories behind your favorite vampires. Who says we only get to celebrate vampires at Halloween?

And yet, scholars have largely turned a blind eye to one lingering question: Which of these authors were, themselves, nosferatu? Was night-stalking Lord Byron
secretly a daywalker? Could it be that Anne Rice’s innate understanding of deathless sexiness is itself a result of her fellowship in the camp of the undead?

Until we understand which writers have been secretly propagandizing for Dracular overlords, our understanding of the place of vampires in anglophone culture
will never be fully complete.
</item 75>

<item 76>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—JL

20. Chants of Sennaar

Developer: Rundisc

Where to play: Nintendo Switch, PlayStation 4, Windows PC, and Xbox One

In Chants of Sennaar you climb up a kind of Tower of Babel, and your job is to decipher the distinct language each group of people speaks, eventually aiming to
translate fluidly from one language to another. You do so by matching pictographic symbols — the characters for each language — to images of the nouns, actions,
and concepts in a large dictionary-style book.

It sounds complicated, but it’s wonderfully fun because the mechanics are so simple. You explore the isometric world — which is rendered in a gorgeous
cel-shaded style — witnessing interactions between cultures and attempting to puzzle out the meaning of their written words. The game parcels out words for you
to assign meaning to in little packets, to avoid overwhelming you. By the end, you’ll have translated numerous languages, and scaled your way to the top of the
tower. Perhaps you’ll even change the tower itself.
</item 76>

<item 77>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
You play as a broody Clive Rosfield, a young man whose life’s work is to protect his little brother, Joshua. The story begins when Clive’s life takes a turn for
the worse and he vows to destroy the monster who ruined his and his family’s legacy.

Developed by Creative Business Unit III, Square Enix’s internal team behind the MMORPG Final Fantasy 14, 16 leans into patchwork territories of fantasy genre
fare. There is palace intrigue, a whole lot of sex, and endless war between nations. But the developers then sprinkle in Final Fantasy elements like mother
crystals, dazzling kaiju fights between summons (known as Eikons in this iteration), and of course, Chocobos.

The quality of the story in this long and linear character-driven RPG waxes and wanes, but the action combat is among the best I’ve ever played. The gameplay
grips you from the very beginning as Clive smoothly dashes, parries, and swings his giant sword and varied magic with a dazzling amount of style.
</item 77>

<item 78>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The game also sheds its predecessor’s odious Spider-Cop bits in favor of establishing a deeper connection between the Spider-Mans and their city, giving the
game room to breathe while further immersing you in its heroes’ worlds and struggles.

And then there’s the gameplay. For my money, there are few experiences more enjoyable in gaming than swinging around New York City with Spider-Man. It was fun
in the early 2000s, and it’s still fun now. And with a bigger map to explore (and wings to fly, if you so choose to use them), there’s basically no limit to the
fun you can get up to. The deployment of two protagonists is seamless — Peter and Miles play differently, bringing their own stories and desires to the table —
and switching between them is effortless. It’s also the rare open-world game that doesn’t feel bloated with missions and side quests, instead leaving you
wanting more Spider adventures. —PV

7.
</item 78>

<item 79>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
One of those 10,000 Rakes definitely belongs to Royce Lewis. 💪 pic.twitter.com/3gRJp2i8eb — MLB (@MLB) October 3, 2023

Our take: We have only a small sample of postseason games at Target Field, but, boy, Twins fans can make some noise. After Game 1 against Toronto, Twins manager
Rocco Baldelli said, “I thought the place was going to split open and melt, honestly. It was out of this universe out there on the field. The fans took over the
game. They helped us win today.” The next day, they did it again.

Advertisement

Early in Game 2, Correa told Sonny Gray that, because of the crowd noise, Jays baserunners couldn’t hear their third-base coach screaming, “BACK!” So, with
Guerrero on second and a full count to Bo Bichette, they executed a perfect timing pick.
</item 79>

<item 80>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
What’s more, Larian had two years of player feedback with which to build its new masterpiece. That masterpiece launched in August, and it garnered the kind of
attention that catapults a game from niche interests to widespread acclaim.

Old CRPG fans finally came home, the Mass Effect generation has discovered a new kind of RPG to sink its teeth into, and D&D fans have found an exceptional
digital version of their beloved tabletop game to play alone or with their party in co-op.

Like a good DM, Baldur’s Gate 3 teases out story beats that feel personal to you, cleverly luring you into experiences that seem like they wouldn’t make sense
in anyone else’s campaign. Every act delivers quiet, memorable character moments and anxiety-inducing battles. All of these situations feel organic, allowing
Baldur’s Gate 3 to replicate, perhaps as closely as a video game can, Dungeons & Dragons’ best feature: the intoxicating sense that anything could happen at any
moment.
</item 80>

<item 81>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
I got caught up in the heat of the moment."

"What I said was completely off base and I've had an interesting 24 hours and I've learned a lot more and I apologize to Jake. I talked to him already, I
apologized to him privately and I also apologize to him publicly," Rizzo said. "They do a good job in the clubhouse and I was completely inaccurate in my
assessment of what went down there."

Alanna Rizzo gave "a very heartfelt apology to Jake Mintz and the entire BBWAA" pic.twitter.com/rT9tjSWLY3 — Awful Announcing (@awfulannouncing) October 13,
2023

Mintz, though calling her initial comments "incredibly hurtful/uncalled for," wrote on X saying that he was "glad she apologized," and asked people not to toss
hate to her.
</item 81>

<item 82>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
One of the Miles-specific side quests involves learning about the artistic history of Harlem, and it entwines the real-life histories of Josephine Baker,
Langston Hughes, and Charlie Parker with the fictional history of SHIELD and other Marvel properties. An early mission shows off Insomniac’s technical chops
with a section that cleverly borrows a trick from Ratchet & Clank: Rift Apart. Of the two Spider-Mans, it’s Miles who feels more alive with the possibility to
break free of expectation and canon.

[Ed. note: Spoilers follow for Spider-Man: Across the Spider-Verse.]

The recent Spider-Man: Across the Spider-Verse has an obsession with “canon events,” pivotal moments in a Spider-Man’s life that the film’s villain, Spider-Man
2099, is dedicated to maintaining across the multiverse.
</item 82>

<item 83>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
While the playing habits of a Twitch streamer don’t represent the average player by any means, content creators certainly shape public opinion about games.

“Negative sentiment is very profitable,” says the AAA game developer, who noted that social media platforms are built to highlight popular — and sometimes
caustic — content. This dynamic sets up an inherent tension between game developers and hardcore players like Twitch streamers, who are not the main audience of
most games but are often some of the most visible players.

“I don’t at all think that streamers are embodying some typical player behaviors, like ‘Brandon,’” the developer continues, posing a name for an imaginary
everyday consumer. “There’s definitely a model of consumers nowadays who probably parrot those sentiments. They did not arrive at those conclusions themselves.”

“There is inherently [a] vicious and negative cycle for everyone involved, for how live service interacts with Twitch as a medium”

The misinformation will vary, but Twitch both shapes that discourse while also becoming the very metric that people use to make sense of a game’s situation.
</item 83>

<item 84>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
When the Phillies beat the Braves in Game 4 to eliminate Atlanta and advance to the NLCS, players in the Philadelphia clubhouse celebrated by repeating, "Atta
boy Harper."

While the comments became a spark for the Phillies, it also became a subject of controversy. Players, including the Blue Jays' Kevin Gausman, decried the
situation as a mistake by Mintz to report the comments, which were said out loud in the locker room rather than said in an interview setting. Others defended
the situation, saying that anything in the clubhouse is open to be reported on.

MORE: Why Arcia's comments motivated Harper after NLDS Game 2

But perhaps the biggest backlash came from MLB Network's Alanna Rizzo, who blasted Mintz on "High Heat" for reporting on the incident. Following a BBWAA
statement, Rizzo both publicly and privately apologized to Mintz for her reaction.

Here's what you need to know about the controversy.
</item 84>

<item 85>
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
</item 85>

<item 86>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
She said that reporters abstaining from writing on his comments would be akin to "protecting players from themselves."

She also went on to say that what Arcia said wasn't a slur or hateful, and added the only reason anyone was concerned about it was because "they know it would
motivate Harper to beat [the Braves]. Which implies the Braves do not feel they can handle a motivated Bryce Harper and are willing to suggest that is why they
lost the other day."

"I didn't say something sooner because I thought this was overblown," Janes continued. "And frankly, I thought it wouldn't help. But I am genuinely disgusted by
the way some media members are acting like a reporter was in the wrong for this. He wasn't.

"And if the Braves let this be the story of their series, that's on them. Win and it goes away. Period."
</item 86>

<item 87>
The best apps and tools for managing your money online (The Verge, 2023-11-12T13:00:00+00:00)
– Nick

“Castro, my favorite podcast app. I appreciate the way that it treats episodes more like emails so I can queue, save for later, or delete them individually.
After using it, I can’t move to any other podcast app.” – Mike

“Found the show Detroiters. Really fun 30-minute comedy starring Tim Robinson and Sam Richardson, with their sensibilities.” – Travi

“I’m playing a game called Chants of Sennaar right now, and it is unlike any other puzzle / logic game I have played. You wake up in a tower with different
classes of people on each level, and they all speak different languages (and obviously, you don’t know any of them). You try to learn each language through
context from people or environment. It is super chill but challenging.” – Bahadir

“I stumbled upon the Beli app for tracking and discovering restaurants, and it is by far the best option for that sort of thing, far better than a crowded
Google Maps or noisy FourSquare.
</item 87>

<item 88>
Super Mario Bros. Wonder review: The joy of pure imagination (Engadget, 2023-10-20T14:11:08+00:00)
It’s a mechanic that feels inspired by recent roguelikes, such as Hades, and it does a great job of adding customizability and replayability.

Have you ever thought about what it would be like to hear Piranha Plants serenade you? Screenshot by Sam Rutherford/Engadget

However, the biggest twist in the game is the Wonder Flowers themselves. In addition to the one Bowser stole, there’s also one hidden in almost every stage. And
if you find it, you better be prepared for the unexpected. Touching them transports you to an alternate dimension where the rules of the Marioverse have been
completely rewritten. In one level Super Stars rain down from the sky, while in another you might be transformed into a giant slime. You may also run into
Wonder Flowers which converts the game from a side scroller into a top-down 2D maze or suddenly find yourself the target in a shooting gallery.

Almost every flower is different and the possibilities are seemingly endless.
</item 88>

<item 89>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
"Thanks to all the reporters, journos, baseball twitter weirdos, acquaintances and friends who had my back over the last 48 hours," Mintz wrote. "And also, of
course, the BBWAA. Community feels good, humbled to be a part of this one. Now, let's enjoy some ball."
</item 89>

<item 90>
The best surprise vampires in things you did not expect to see vampires (Polygon, 2023-11-15T18:02:00+00:00)
We're spending a week diving deep into the stories behind your favorite vampires. Who says we only get to celebrate vampires at Halloween?

You know what’s cooler than a vampire story? A story that’s not a vampire story, but surprise — it has vampires.

The surprise vampire is a rich tradition, unfolding across many different media intended for all kinds of audiences. Sometimes it is a prominent part of a story
that’s made richer for the unexpected presence of a bloodsucker.

Sometimes it is subtle, an easily missable detail that rewards only the most observant. In the best version, the vampires are always there, in the background,
waiting for you to discover them — just like in real life.

Here are some of our favorite examples.

Baldur’s Gate 3: Astarion

If Baldur’s Gate 3 was your introduction to the Forgotten Realms of Dungeons & Dragons, you might be shocked to find out about a lot of stuff that’s tucked away
in the world’s most famous tabletop role-playing game.
</item 90>

<item 91>
Pokémon is no longer just a game — it’s a lifestyle (The Verge, 2023-11-23T18:00:00+00:00)
Illustration by Sarah Oh / The Verge Read next: Pokémon Sleep helped me catch ’em all — all the z’s, that is

Much of the opening episode of the series plays out as a nostalgic parallel between the trials of working life and the journey to become a Pokémon Master. The
childhood friend who works for another company is your rival. The product presentation to an investment CEO with the company’s status on the line is a gym
battle.

She rediscovers Pokémon in a box of items sent from her mother and begins to replay this formative adventure; there are clips of the game or shots where we see
the joy on the faces of both child and adult Madoka from the dim glow of the Game Boy screen. Pokémon love even intertwines with the music scoring the series: a
relaxing bar plays a slow jazz-infused rendition of the Pokémon Center theme, just like how a synth rendition of Pallet Town marks the start of Madoka’s
journey.
</item 91>

<item 92>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
I’ve rolled for every character, grinded out all of said characters, 100%-ed the exploration, completed every quest — did all the stuff that would burn anyone
out. Honkai: Star Rail is Hoyoverse’s answer to the Genshin Impact burnout.

The slower-paced turn-based combat, the ability to auto-battle (if your teams are strong enough), and the smaller maps make it the perfect daily game for me to
funnel my interest into without feeling exhausted. The characters are fun, flashy, and easy to latch onto. Should I main the sneaky but elegant Kafka, who uses
damage-over-time skills to whittle her opponents away? Or should I use the brooding Blade, who unleashes huge attacks at the cost of his HP? Ah, I guess I’ll
let my wallet decide — it is a gacha game, after all. —JL

20.
</item 92>

<item 93>
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
</item 93>

<item 94>
Pokémon is no longer just a game — it’s a lifestyle (The Verge, 2023-11-23T18:00:00+00:00)
As the show develops, the lessons from the game form the basis of a pseudo-episodic tale about Madoka’s life at the company. The struggle to connect with a new
client is paralleled by the way pokémon won’t listen to you until you’ve proven your worth with gym badges. Learning and improving on the job is the experience
that turns a flailing, floundering Magikarp into a job-leading Gyarados. And so on.

The show can come across a bit silly in places, and it struggles with communicating itself by making its Pokémon overlay at times difficult to take seriously.
But there’s heart: there are plenty of touches that represent a series made with love, such as how characters have their own sprites integrated into the
gameplay of Pokémon Red.

With PokéTsume, Pokémon isn’t just a virtual journey to become a champion. It’s a lesson we take with us into our daily lives long after our adventure is at an
end.
</item 94>

<item 95>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It actually might be impossible to be angry while playing this game. Hearing the little Pikmin hum as they carry a huge peach (that they call a “mock bottom”)
across the map is enough to melt anyone’s heart. In fact, I was pretty sad when the game ended. I could command these little guys and my puppy friend to collect
things forever.

All told, the game is beautiful, and, thanks to its quasi real-world setting, it makes me see my own world differently — there’s something special to be found
everywhere, even in the tiniest corners. Maybe all my little trinkets go missing because a little guy needs it to return home. He can have it.

Even if you’ve never played a Pikmin game before, this game is good, and it’s the perfect place to start. (And then you can play all the rest of them, which
have been conveniently ported to the Switch!) —JL

Related Pikmin 4 will turn you into an obsessive collector

4. Super Mario Bros.
</item 95>

<item 96>
Super Mario Bros. Wonder review: The joy of pure imagination (Engadget, 2023-10-20T14:11:08+00:00)
Since they’re meant for both experienced gamers and children who might be new to the franchise, they often locked more challenging levels behind the story’s
completion. This meant the beginning of the game could be a slog as you rushed through easier levels to get to the more challenging bits. But with the addition
of difficulty ratings and hidden paths that lead to 5-star levels long before you finish the game, there isn’t the same rush to get to the “good stuff.” And I
should say that 4- and particularly 5-star levels are balanced just right. Completing one feels like a rewarding challenge, without crossing over into
punishing.

I do have some very minor complaints about things like mummy enemies (pulling their wraps feels a bit tedious) and I fear the Elephant Suit might be a bit
overpowered in comparison to the Drill Hat. The 5-star levels and rhythm stages are also so good I wish there were more of them, I just can’t get enough.
</item 96>

<item 97>
Pokémon is no longer just a game — it’s a lifestyle (The Verge, 2023-11-23T18:00:00+00:00)
Within all that were families, often two or three generations, sharing in a mutual love of Pokémon.

The results speak for themselves. In recent years The Pokémon Company has recorded record profits and revenue, while the games, anime, and TCG are more popular
than ever, with both new mainline Switch entries selling over 20 million copies. The phenomenon of growing out of Pokémon is no longer a concern. Even if young
kids stop playing the video games as they grow, that doesn’t mean they won’t still engage with this world via other means. Nor does it mean they won’t introduce
their own children to the series in the future.

PokéTsume is a series that’s only possible because of a reinvention of what Pokémon can be. While, in reality, it’s a mostly light, inoffensive promotional tool
to sell games (it’s no coincidence the second episode has a scene of a young kid playing the latest entries in the series on his Nintendo Switch), it represents
everything the franchise has aimed to be in recent years.
</item 97>

<item 98>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
But that would also be underselling the game, and understating things quite a lot.

Diving into the mysterious Blue Hole, Dave spends the first two quarters of his day swimming deeper into the colorful abyss, discovering both sea life and a
story that’s equally absurd and earnest. When you’re not picking up sea urchins or spearfishing sharks, Dave is assisting the rest of Dave the Diver’s cast of
characters — his sushi business partners, a community of seafolk, an anime-obsessed weapons expert, and a pair of dolphins. At night, Dave slings sushi and
pours drinks at the restaurant, frantically running back and forth between clearing dishes, delivering sushi, and refilling the freshly ground wasabi. Between
all that, Dave’s harvesting rice and vegetables on a farm, curating a hatchery, racing seahorses with mermaids, and taking down a suspicious group masquerading
as environmental activists. Somehow, there’s even a well-done rhythm video game — starring one of those anime idols that the arms dealer loves — that makes
perfect sense.
</item 98>

<item 99>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Peter’s plot involves a retelling of the Venom origin story, with beats that will be recognizable to those even passingly familiar with the Spider-Man mythos.
If Spider-Man 2 is a game of retellings, Peter’s story clings most closely to the canon. If you’ve seen Sam Raimi’s Spider-Man 3, you’ll know the gist of what
to expect when the symbiote finds its way to Peter. It isn’t until the late game that Peter’s story starts pushing more strongly against our canonical memories
and into something more unique to Insomniac.

Miles’ story takes more liberties. Hailey Cooper, Miles’ love interest from his solo game, returns here, taking more of an MJ-esque role and carving out her own
identity separate from Miles’. American Sign Language is featured heavily throughout Miles’ missions because of Hailey’s presence, feeling as natural and
integral to the story as anything from the canon.
</item 99>

<item 100>
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
</item 100>
=== END ITEM id=1eaa21009070277b ===
