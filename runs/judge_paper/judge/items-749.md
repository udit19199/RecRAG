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

=== ITEM id=677cc0f27ed06a58 ===
Gold sentences and facts:
Sentence 0: “She’s got the Eras Tour booked all through next year, and she’d rather die than cancel a show.
  Fact 0.0: The quoted woman (not named in this sentence) has the Eras Tour booked all through next year.
  Fact 0.1: The quoted woman would rather die than cancel a show.
Sentence 1: "It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something really major and traumatic happen to me and they don't know about it," she told NPR.
  Fact 1.0: The speaker (a woman not named in this sentence) said that it was almost like a very strange, subtle clue to the media.
  Fact 1.1: The speaker said the media do not know everything that happened in that relationship.
  Fact 1.2: The speaker said she can have something really major and traumatic happen to her without the media knowing about it.
  Fact 1.3: The speaker told NPR this.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has now shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.1: Swift has a new relationship with Kelce.

Retrieved items:
<item 1>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Image source, Getty Images Image caption, 1989 was seen as a risk by Taylor's record label, but ended up becoming her biggest-selling album to date

5) All You Had To Do Was Stay

Traditionally, track five on a Taylor Swift album is the record's most raw and vulnerable moment - but All You Had To Do Was Stay is no Dear John or All Too
Well.

Instead, it's a big disinterested "nope" to an ex who comes back begging for a second chance.

The song came to Taylor in a dream where her ex turned up at the door and, instead of giving him a piece of her mind, all that came out was a high-pitched plea
to "stay".

When she woke, she recorded the sound into her phone and decided to base a song around it. Only this time, she got to say everything she wanted.
</item 1>

<item 2>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“I think some people think that they saw our first date at that game? We would never be psychotic enough to hard launch a first date,” Swift joked.

Back in April, it was revealed that Swift and British actor Joe Alwyn had broken up after six years of dating. The former couple were notoriously private, as
the Conversations with Friends star previously expressed he didn’t want to discuss their relationship because it “feeds into a weird part of the culture”.

Now, Swift has shared that she has nothing to hide in her new relationship with Kelce. “When you say a relationship is public, that means I’m going to see him
do what he loves, we’re showing up for each other, other people are there and we don’t care,” she said. “The opposite of that is you have to go to an extreme
amount of effort to make sure no one knows that you’re seeing someone. And we’re just proud of each other.”
</item 2>

<item 3>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"Yeah, my personal life -- that's not so personal. I did this to myself Jason, I know this. What's real is that you know it is my personal life. ... She's not
in the media as much as I am doing this show every single week."

Sept. 27: Jerry Jones comments on the Swift effect

Swift is getting the owners talking, too.

"It is hard to move that needle dramatically on a great feeling, a great attraction in the NFL," Jones said Wednesday on 105.3FM The Fan in Dallas. "We get a
lot of eye balls. She blew it off the charts. It is amazing. It makes you smile. But it just shows the depth of her following, especially with her core
constituency. We all, I think, felt the same way when she's sitting up there with [Travis Kelce's] mom and she's going nuts when he's playing and those hearts
are popping out everywhere.
</item 3>

<item 4>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Swift’s name began popping up everywhere on the podcast, including during conversations about... facial hair?

In August, the brothers discussed Kelce’s moustache, which he grows every year in honour of the Chief’s head coach, Andy Reid. When asked if he knew what Swift
thought of the ’stache, he remained cheekily confident. “We’re not going to bring up Taylor Swift in this episode,” he says, “But something tells me she’s going
to like it.” There was a clear suggestion Kelce had made contact with Swift, and was even going to meet her. Shortly after that remark, Kelce took a week off
training to visit New York City, according to Vulture. Coincidentally, this happened to be while Swift was re-recording her 1989 tracks in the city. Welcome to
New York

The suspicion the two were actually spending time together in New York was seemingly confirmed in September when it was reported they had been “quietly hanging
out” for months.
</item 4>

<item 5>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Could they have been driving off into the sunset? Of course, none of this actually confirms Swift and Kelce are actually dating. But regardless of whether it’s
a friendship or a romantic relationship, it has sent shockwaves through two massive cultural franchises, expanding the NFL’s audience into uncharted territory.
It’s a cultural pairing on par with – and perhaps even bigger than – Kim Kardashian and Kanye West.
</item 5>

<item 6>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Those search engine clicks gotta be lucrative, right?

Image source, Getty Images Image caption, Many of 1989's songs were supposedly inspired by Taylor's relationship with Harry Styles

3) Style

An epic aural painting of windswept coastlines and propulsive wanderlust, Style is (supposedly) named for Harry Styles, with whom Taylor had a brief
relationship in 2012.

Except she says it was inspired by fashion, and how certain items like a black dress or red lipstick will "never go out of style". That prompted her to write
about the people who'll always be part of your life, even after the flame of romantic love wanes (all of which sounds like a classic piece of deflection to a
probing journalist).

Whichever interpretation you prefer, the song marks a shift in Swift's psychology, as she briefly admits to having been unfaithful.

"I would never have said anything like that on a previous album," she told Ryan Seacrest.
</item 6>

<item 7>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
It's certainly not her finest work - a vendetta disguised as a pop song, with a ham-fisted chorus that lacks her usual melodic instincts.

Swift hinted in a Rolling Stone interview that the track was about a high-profile female musician, saying the person did something "so horrible" that made them
"straight-up enemies."

Although she tried not to identify the target, it became common knowledge that Katy Perry was in the crosshairs, after allegedly "stealing" Taylor's backing
dancers just before a tour.

Perry confirmed the situation on James Corden's Carpool Karaoke, adding that she "tried to talk to [Swift] about it and she wouldn't speak to me."

The pop divas eventually reconciled in the 2019 video for You Need To Calm Down... but Taylor's still performing Bad Blood on her Eras tour.
</item 7>

<item 8>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
'She dates too many men!") with a defiant "so what?", sounding much less bothered than the girl who'd written Mean as a riposte to a snarky music critic three
years earlier.

Released as 1989's first single, Taylor spent three days shooting a video, codenamed Project Sparrow, that built on the lyrical conceit.

"When I'm at award shows, I have the best time dancing [but] I'm really tall and lanky so I look a little bit like a baby giraffe learning how to walk," she
told a group of young Swifties.

"So all over the internet, it's like,'Taylor Swift looks awkward when she dances'. And as soon as I wrote [Shake It Off] I knew I wanted to do an entire music
video of me awkwardly dancing.

"I'm like, 'Yeah, I do dance awkwardly, but I'm having more fun than you.'"
</item 8>

<item 9>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"I just wrote it down, and closed the book and put it back on my nightstand, like, 'Oh, that was fun'.

"Then all of a sudden in my head, I started hearing this melody - and that's when I realised it was going to be a song."

12) I Know Places

Continuing the theme of thwarted love, I Know Places was fuelled by Taylor's increasing anxiety over holding down relationships in the spotlight.

"I was like, 'No-one's gonna sign up for this. There are just too many cameras pointed at me, there are too many ridiculous elaborations on my life'," she told
the Grammy Museum.

"But I decided to write a love song [about] what would I say if I met someone really awesome and they were like, 'Hey, I'm worried about all this attention you
get'.

"So I wrote this song about, 'Hey, I know all these places we could hide. We can outrun them'."
</item 9>

<item 10>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
But that's for another article..

17-21) Tracks from the vault

In 2014, Taylor said she had recorded more than 100 songs for 1989 - so the relative scarcity of vault tracks is intriguing.

Whereas previous re-recordings have featured full discs of bonus material, this album gets just five new additions: Slut!, Say Don't Go, Now That We Don't Talk,
Suburban Legends and Is It Over Now?

As a whole, they feel like dry runs for the songs that made the cut. The melodies aren't as crisp, the lyrics aren't as sharp. Suburban Legends has a
particularly clunky line about letting a partner's indiscretions slide "like a hose on a slippery plastic summer".

Slut! has piqued fans' interest for its title alone - who might she be talking about? No-one, it turns out.
</item 10>

<item 11>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Suddenly, football fans and Swifties were living through this surprising collision. Swift lyrics were infiltrating NFL game commentary, and her fans were
furiously becoming football experts. All that needed to happen was for Swift herself to infiltrate a game, determining once and for all whether there was
anything to these rumours or not. This Is Me Trying

On September 21, Kelce announced on The Pat McAfee Show he had invited Swift to watch one of his games. “I told her, ‘I’ve seen you rock the stage in Arrowhead,
you might have to come see me rock a stage at Arrowhead and we can see which one’s a little more lit,’” he said. “We’ll see what happens in the near future.”
Were they finally about to hard-launch their friendship/potential romance to the world, laying all speculation to rest? Love Story This finally brings us to the
fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No. 87 with gusto.
</item 11>

<item 12>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
- Enchanted

Mahomes said he had not met Swift when asked in the postgame press conference, but he did meet her at the afterparty.

"Yea I met her. She's really cool. Good people," Mahomes said, via KSHB 41 in Kansas City. "But like Trav said, imma let them have their privacy and just keep
it moving."

Sept. 27: Kelce breaks his silence

"And they said speak now." -Speak Now

Kelce addressed Swift's attendance on his podcast, saying, "To see the slow-motion chest bumps, to see the high-fives with Mom, to see how Chiefs Kingdom was
all excited that she was there -- that s--- was absolutely hysterical. It's definitely a game I'll remember, that's for damn sure."

He is not going to speak too much more about the relationship, noting that he wants "to respect both of our lives."

"Yeah, my personal life -- that's not so personal.
</item 12>

<item 13>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
It's so weird to look back on that - you've got heaping piles of discarded songs that I really loved when I wrote them and I still love now.

"I just need each album to have its own sound and if it doesn't fit that sound, it doesn't matter if it's a great song to me."
</item 13>

<item 14>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
24: Swift mingles with Donna Kelce



"Talkin' over a football team." - Mean

Swift already seems to be making friends with the family. Swift was sitting next to Donna in the box and the two were seen talking and laughing as the game went
along.

Sept. 24: Postgame interviews turn into Swiftie talk

The postgame press conference seemed more about Swift and Kelce than it did about the actual game. The media wanted any insight it could get into the
relationship and how Swift ended up at Arrowhead.

"He told me at the last minute," Mahomes said when asked if he knew Swift was coming. "There's some things with Trav where he kind of just says it and you don't
know if it's true or not. He says it so calmly. ... Friday he was like, 'Yeah, I think she's coming to the game this weekend,' and then moved about his
business."
</item 14>

<item 15>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
In interviews, Taylor delighted in revealing this was based on a true story.

"It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something
really major and traumatic happen to me and they don't know about it," she told NPR.

In her Rolling Stone cover story, Taylor said the crash actually occurred on a snowmobile "with an ex who lost control and wrecked it so badly that she saw her
life flash before her eyes".

That prompted speculation the track was also about Styles: The couple had gone skiing with Justin Bieber and Selena Gomez in late 2012.

The lyrics also reference their identical aeroplane necklaces and a December break-up, which ties in with the "Sad Taylor on a boat" meme.

But the singer steadfastly refuses to name names: "The fact that I've never confirmed who those songs are about makes me feel like there is still one card I'm
holding," she told GQ.
</item 15>

<item 16>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
The decision came after one of the singer’s fans, 23-year-old Ana Clara Benevides, died from a cardiac arrest during Swift’s first performance in the country.
Swift explained to Time that “part of my identity as a human” is putting on a show for her fans, no matter the circumstances. “I know I’m going on that stage
whether I’m sick, injured, heartbroken, uncomfortable, or stressed,” she said. “That’s part of my identity as a human being now. If someone buys a ticket to my
show, I’m going to play it unless we have some sort of force majeure.”

Perhaps it was her decision to re-record her music catalogue that catapulted Swift into such success. When music mogul Scooter Braun aquired Swift’s longtime
label, Big Machine Records, in 2019, he also gained the rights to the master recordings of Swift’s first six studio albums. In November 2020, Braun sold the
masters to an investment fund in a deal that was believed to be over $300 million.
</item 16>

<item 17>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"A farewell to twang," the New York Times called it.

Taylor knew change was necessary. "I don't have the option of making music that sounds just like what I've done before," she told Popcrush. "People will see
right through it. They'll see that I was lazy".

Her label begged her to add three country-leaning songs to the album, but she stood her ground.

"At a certain point, if you chase two rabbits, you lose them both."

Welcome To New York itself is the sound of creative and personal freedom. Taylor moved to the city in 2014 and was amazed to discover she could go grocery
shopping largely incognito.

"My friends and people who know me have commented that I'm physically different since being here," she told NYC radio station WKTU. "That song is about the
endless hopefulness and endless possibilities that New York seems to present," she added in an interview with Access Hollywood.

All proceeds from the track are donated to New York City's Public Schools.
</item 17>

<item 18>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has officially been named Time’s 2023 Person of the Year.

On Wednesday 6 December, the magazine announced Swift had earned the honour after an “editors’ assessment of the individual who most shaped the headlines over
the previous 12 months, for better or for worse”.

The recognition of Swift comes on the heels of a whirlwind year for the popstar, whose Eras Tour dominated ticket sales in 2023.
</item 18>

<item 19>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She specified that she isn’t trying to hide her relationship from the public eye.

“When you say a relationship is public, that means I’m going to see him do what he loves, we’re showing up for each other, other people are there and we don’t
care,” she said. “The opposite of that is you have to go to an extreme amount of effort to make sure no one knows that you’re seeing someone. And we’re just
proud of each other.”

During the interview with Time, the Grammy winner addressed how her appearances at Kelce’s games are heavily publicised. So far, she’s attended five of Kelce’s
games, including one on 3 December, and she’s often seated in the Kelce family suite during the events.

“There’s a camera, like, a half-mile away, and you don’t know where it is, and you have no idea when the camera is putting you in the broadcast, so I don’t know
if I’m being shown 17 times or once,” she said.
</item 19>

<item 20>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.
</item 20>

<item 21>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
ADVERTISEMENT Article continues below this ad

How can you not love Taylor Swift? Most of the 12 Grammy Awardee’s songs are about that warm fuzzy feeling and chemical reaction that we call love. And her
songs have earned the support of 103.7992 million Swifties in the world. Yeah, we don’t have a round figure for that. We have the exact figure up to four
decimals. That’s how much data we have on her. Football has 184 million fans. The league is able to generate a revenue of $18 billion from 184 million. If it
adds 104 (after rounding 103.7992) million to that number, guess how much revenue it will generate? $28 billion. And just like that, their target will be met.
</item 21>

<item 22>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
No-one, it turns out.

The song is thematically similar to Blank Space, commenting on the media's portrayal of Taylor's relationships: "But if I'm all dressed up / They might as well
be lookin' at us / And if they call me a slut / You know it might be worth it for once."

The best of the new tracks is Is It Over Now. An angrier take on 1989's philosophical break-up songs, it calls back to the Harry Styles snowmobile incident (see
also, Out Of The Woods) then, deliciously, references the "Sad Taylor Boat" meme - a photo of Taylor fleeing a vacation alone after the couple split up.

She goes on to call an ex a "lying traitor", accuses them of parading their new relationships in public, and noting that every new girlfriend looks like her.

"If she's got blue eyes, I will surmise that you'll probably date her."
</item 22>

<item 23>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But at the same time, they're overdoing it a little bit."

Jason also chimed in on the matter, saying, "I just think the NFL is not used to celebrities coming to the games. Like basketball has it all figured out.
They're all courtside, they're sitting there. They show 'em once or twice, but then they get back to the game."

Oct. 12: Swift attends Chiefs-Broncos

"Now it's too late for you and your white horse *Bronco, to come around" - White Horse

One day after her movie premiere, Swift is in the house for the Chiefs' "Thursday Night Football" divisional matchup against the Broncos. The game is taking
place at Arrowhead Stadium, where Swift has performed herself and attended a game before. According to Al Michaels, the game will have minimal Swift coverage,
though so far there has already been a mention of the superstar.

Oct.
</item 23>

<item 24>
Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral (The Independent - Life and Style, 2023-11-25T14:50:53+00:00)
In an interview withWSJ Magazine published on 20 November, Kelce described how much he admires Swift’s skills as a musician. “I’ve never been a man of words.
Being around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

Although he acknowledged the public attention his personal life has received because he’s “never dated anyone with that kind of aura around them,” Kelce
specified that he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life.

“When she acts like that I better not be the one acting all strange.”
</item 24>

<item 25>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
This was the first, albeit tentative, confirmation from the media rather than from amateur Swiftie sleuths on social media. So, it was only a matter of time
before it entered the cultural zeitgeist. Blank Space Which it did, very quickly. On September 17, Kelce scored a touchdown against the Jaguars, triggering CBS
announcer Ian Eagle to reference Swift’s 2014 hit, Blank Space.

“Kelce finds a blank space for the score,” Eagle said, poking fun at Kelce’s newfound Swiftie status. And Eagle wasn’t the only one, with NFL Network’s Rich
Eisen including a slew of lyric references during a preshow, including “shake it off”, “anti-hero” and “all too well”. Kelce even acknowledged the
tongue-in-cheek banter, commenting on a montage video of the preshow that Eisen posted to Instagram. “Well played Rich … Well played,” he wrote. Suddenly,
football fans and Swifties were living through this surprising collision.
</item 25>

<item 26>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She explained that she doesn’t necessarily pay attention to the cameras and fans around her in those stadiums because she is watching her boyfriend play.

(Denver Post via Getty Images)

“I’m just there to support Travis,” she said. “I have no awareness of if I’m being shown too much and pissing off a few dads, Brads, and Chads.”

When discussing how many Swifties have now started watching NFL games since she started dating Kelce, she quipped about how much she likes the sport. “Football
is awesome, it turns out,” she joked. “I’ve been missing out my whole life.”

Swift’s effect on the NFL didn’t go unnoticed in October, as one game between the Chiefs and New York Jets - which she attended with some of her famous friends
- became the most-watched Sunday NFL show since the Super Bowl LVII in February 2023.
</item 26>

<item 27>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
This is the biggest and riskiest part of the project. While earlier remakes were largely made with a live band, 1989's pop landscapes are full of squelchy,
processed synths and treated vocals.

Fans can now find out how it has turned out - and hear five new tracks from the star's vault, expanding on the themes and relationships she explored on the
original.

Here's a guide to all the songs and what Taylor has said about them.

1) Welcome To New York

"It's a new soundtrack, I could dance to this beat forever more."

We open with a mission statement: The old Taylor can't come to the phone right now, she's dead.

After building her reputation in Nashville, 1989 jettisons the banjos in favour of insistent, needling beats and tales of bohemian nightlife. "A farewell to
twang," the New York Times called it.

Taylor knew change was necessary.
</item 27>

<item 28>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Like, through thick and thin, sickness and health love," she told MTV.

Drawing directly on conversations she'd had with Lena, the song blends scenes of everyday life (burnt toast, borrowed shirts) with moments of incandescent
tenderness (kisses on sidewalks and late night heart-to-hearts).

In an interview with Elle, Taylor talked wistfully about the couple's ease around each other.

"I've never had that," she said. "[But] now I have more of a grasp on the fact that when you're in a state of infatuation and you think everything that person
does is perfect, it then - if you're lucky - morphs into a real relationship when you see that that person is not in fact perfect, but you still want to see
them every day."
</item 28>

<item 29>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Image caption, Taylor was one of the main stage headliners at Radio 1's Big Weekend in 2015

7) I Wish You Would

One of Taylor's many 2am songs, I Wish You Would, is powered by a choppy, unrelenting guitar line that mimics her thoughts as they race uncontrollably in the
middle of the night.

She's pining for an ex, absent-mindedly driving down their street and wishing she'd never hung up the phone or broken things off.

But in real life, the story was reversed. Taylor told Rolling Stone the song was "about an ex who bought a house two blocks from hers"; while a hidden message
in the CD booklet said: "He drove past her street each night."

The mystery deepens.

8) Bad Blood

Bad Blood has the dubious distinction of taking last place in Rolling Stone's ranking of all 237 Taylor Swift songs.
</item 29>

<item 30>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
The “All Too Well” singer’s interview with Time marks her first time publicly speaking out about Kelce, who she’s been romantically linked to since this summer,
when the football tight end first confessed to attempting to give her his number.

While Swift has attended a few of Kelce’s games since their relationship started, he’s also gone on to support her career. In November, he made his way to
Buenos Aires, Argentina, to attend one of her Eras Tour shows. During the concert, Swift gave a shout-out to the Chiefs player when she changed her “Karma”
lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang, letting out a laugh as fans in the audience
screamed in response.

After the show, the “Style” singer was seen running into Kelce’s arms and kissing him, before he whisked her away backstage.
</item 30>

<item 31>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Some found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting as such.

"We frequently change our bios and profile imagery based on what's happening in and around our games, as well as culturally," the NFL wrote in a statement. "The
Taylor Swift and Travis Kelce news has been a pop cultural moment we've leaned into in real time, as it's an intersection of sport and entertainment, and we've
seen an incredible amount of positivity around the sport."

Oct. 6: Travis and Jason think the NFL is doing too much

The NFL is really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.

"I think it's fun when they show who's at the game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what
you're watching. But at the same time, they're overdoing it a little bit."
</item 31>

<item 32>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"If she's got blue eyes, I will surmise that you'll probably date her."

The fact that all the new songs were co-written and produced by Jack Antonoff suggests there may have been rights issues over the tracks she wrote with Max
Martin - who has not been involved with the new re-recordings.

That means the vault tracks are all bathed in the dreamy, soft-focus atmospherics of Out Of The Woods, rather than the upfront pop of Shake It Off.

Speaking in 2015, Taylor admitted she'd been brutal about pruning songs from 1989's original tracklist.

"There were so many songs I wrote for this album that were really good, but if I felt in any way that they would have belonged on my last album, Red, they got
cut," she told New York radio's The Todd Show in the Morning.

"There are really, really good songs on the cutting room floor because they weren't innovative enough or new enough.
</item 32>

<item 33>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"I would never have said anything like that on a previous album," she told Ryan Seacrest.

"My previous albums have always been, 'I was right, you were wrong'. What happens when you grow up is you realise that the rules in a relationship are very
blurred. It gets very complicated, very quickly."

4) Out Of The Woods

One of 1989's strengths is how the music reflects the emotional tenor of the words. Here, that means frantic anxiety.

As Taylor and her partner run from forces that seek to destroy them, they're pursued by pounding drums and rushing synths. And there's a desperation to her
voice, as though she's repeating "everything's fine, everything's fine, everything's fine," against evidence to the contrary.

By the middle 8, it seems they've made a clean getaway. Then car brakes are slammed, and it's "twenty stitches in a hospital room".

In interviews, Taylor delighted in revealing this was based on a true story.
</item 33>

<item 34>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter,” Swift told Time. “Yes, if I go out to dinner,
there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends.

“Life is short. Have adventures. Me locking myself away in my house for a lot of years - I’ll never get that time back,” she added. “I’m more trusting now than
I was six years ago.”
</item 34>

<item 35>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Image source, Republic Records Image caption, Artwork for the re-recorded version of 1989 has a beach vibe, in contrast to the Polaroid aesthetics of the
original

16) New Romantics

Proof that even Taylor gets it wrong: New Romantics, one of her most beloved songs, was originally held back as a bonus track.

It's a beast, though. An armour-plated anthem to resilience and power, where Taylor takes all the pain that's come her way and uses it to make herself
bulletproof.

"Heartbreak is the national anthem, we sing it proudly," she declares, while glitterballs shatter around her. "I could build a castle out of all the bricks they
threw at me."

That defiance also acts as a bridge to her next album, Reputation, which dismantled her critics, one-by-one.
</item 35>

<item 36>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“This is the proudest and happiest I’ve ever felt, and the most creatively fulfilled and free I’ve ever been,” Swift told Time. “Ultimately, we can convolute it
all we want, or try to overcomplicate it, but there’s only one question... Are you not entertained?”

Swift was chosen among a strong list of contenders, including Chinese President Xi Jinping, King Charles III, the live-action Barbie movie, and Vladimir Putin.
“While her popularity has grown across the decades, this is the year that Swift, 33, achieved a kind of nuclear fusion: shooting art and commerce together to
release an energy of historic force,” the magazine said about its selection.

This year alone, Swift has broken records with her career-spanning Eras Tour, which is estimated to have grossed over $1.4bn. Her concert film, the Eras Tour
movie, has surpassed $250m at the global box office since its opening weekend in October.
</item 36>

<item 37>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
And we’re just proud of each other.”

Speaking to Time, Swift detailed exactly how much work went into preparing for her Eras Tour - where each concert spans over 180 minutes, with more than 40
songs from at least nine albums and 16 costume changes. “Every day I would run on the treadmill, singing the entire set list out loud,” she told the outlet.
“Fast for fast songs, and a jog or a fast walk for slow songs.”

She also stopped drinking and underwent three months of dance training, something that she admitted is “not my strong suit”. After she plays a run of shows,
Swift said she takes a day to rest and recover before taking the stage once again. “I do not leave my bed except to get food and take it back to my bed and eat
it there,” she revealed. “It’s a dream scenario.”

Most recently, a heat wave in Brazil forced Swift to postpone the second night of her Eras Tour shows in Rio de Janeiro.
</item 37>

<item 38>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Taylor Swift's biggest album 1989 returns with new tracks from the vault Published 27 October

Image source, Getty Images Image caption, Taylor Swift's 1989 tour sold 2.2m tickets and earned more than $250m at the box office

On Friday, Taylor Swift released a new version of 1989 - the biggest-selling album of her career, and the one that definitively turned her into a pop star.

Featuring hits like Shake It Off, Blank Space and Style, it was originally written during the 2013-14 Red Tour, with demos stored on her phone in a folder named
"Sailor Twips".

Awarded a Grammy for album of the year, it has spent 325 weeks in the UK charts.

But now she has re-recorded it as the latest part of an ongoing campaign to regain control of her work, after an investment company bought her master tapes in
2019.

This is the biggest and riskiest part of the project.
</item 38>

<item 39>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
The first link between Swift and Kelce was established in 2016, when Kelce – who helped bring his team to Super Bowl victory in both 2020 and 2023 – played the
classic game “Kiss, Marry, Kill”. Speaking with AfterBuzz TV, Kelce was given the options of Swift, Katy Perry and Ariana Grande. He said he’d ultimately kill
Grande, marry Perry and kiss Swift. That was about six years ago, but it has more recently set tongues wagging as fans wondered how long the plan for them to
meet has been in the works. Cruel Summer Fast-forward seven years, and things started to heat up. In July, speaking on the podcast he shares with his brother,
New Heights, Kelce said he intended to give Swift a friendship bracelet with his number on it during her Eras Tour concert in Kansas City. Unfortunately, the
bracelet never made it to her wrist or his digits to her phone.
</item 39>

<item 40>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Only this time, she got to say everything she wanted.

Image source, Universal Music Group Image caption, The video for Shake It Off was filmed under intense secrecy

6) Shake It Off

Rattling like a bag of spanners, Shake It Off is a 21st Century Hey Mickey, a honking great party record with a chorus that drills so far into your brain you
start seeing lights.

It was the last song recorded for 1989, and Taylor arrived at the studio with a challenge for her co-writers Max Martin and Johan "Shellback" Schuster: "We need
drums that make even the person who's having a terrible night at the wedding [go], 'Oh, it's my song!'"

Like Blank Space, Shake It Off is a response to celebrity takedown culture. Taylor lists all of her supposed flaws ("She's an airhead!' 'She dates too many
men!") with a defiant "so what?
</item 40>

<item 41>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.
</item 41>

<item 42>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 42>

<item 43>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
All proceeds from the track are donated to New York City's Public Schools.

Image source, Getty Images Image caption, The original version of 1989 won three Grammy Awards, including Album of the Year

2) Blank Space

Pretty much the perfect Taylor Swift song, Blank Space takes every horrible rumour that was being spread about her love life, and amplifies it with satirical
relish.

"In the last couple of years the media have had a wonderful fixation on painting me as the psycho serial dater girl," she told the Grammy Museum. "Every article
was like, 'Here's Taylor Swift standing near some guy. WATCH OUT GUY!'

"My first reaction was to be like, 'That's a bummer. This isn't fun for me'.
</item 43>

<item 44>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
How do you not get that? That's called images right there."

Oct. 1: Swift attends Chiefs-Jets

"Welcome to New York, it's been waiting for you," - Welcome to New York

OK, so the Jets technically play in New Jersey, but we're just basing this off their name. Swift went to her second consecutive Chiefs game, showing up with
some famous friends, Blake Lively, Ryan Reynolds and Hugh Jackman. In the week leading up to the game, there were reports that the singer would be at the game.
The Chiefs defeated the Jets 23-20.

Oct. 5: NFL defends itself

Swift coverage is everywhere and the league even changed their social media bio to one of the singer's lyrics. The league's X (formerly Twitter) bio to read,
"NFL (Taylor's Version)," while their Instagram account read, "Chiefs are 2-0 as Swifties."
</item 44>

<item 45>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Travis’ teammates also showed up to the after-party, as well as his mom, Donna Kelce. The party lasted until 2 am.”

The late-night outing came days after the NFL star officially addressed ongoing dating rumours about him and the “Anti-Hero” singer. During an episode of The
Pat McAfee Show on 21 September, he mocked his brother Jason Kelce for joking that he “100 per cent” believed his younger sibling was dating Swift.

“It’s hilarious how much traction this has actually gotten, right now it’s like a game of telephone,” Kelce said. “Where everybody’s whispering in each other’s
ears and hearing random stuff. No one actually knows what’s going on, especially when you got Jason Kelce on live television telling people just both sides - he
doesn’t know, it’s true, this and that.”

He also noted that he was still open to pursuing a relationship with Swift, revealing that he subtly asked if she’d want to attend one of his football games.
</item 45>

<item 46>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Unfortunately, the bracelet never made it to her wrist or his digits to her phone.

“I was disappointed that she doesn’t talk before or after her shows because she has to save her voice for the 44 songs she sings,” he said. “She doesn’t meet
anybody, or at least she didn’t want to meet me, so I took it personal[sic].” At this point, it seems like any attraction between the two was one-sided. Kelce’s
attempts to make proper contact with Swift at the show ultimately failed. Was she just not that into him? But Kelce, exhibiting the tenacity typical of a
professional sportsman, remained determined to get her attention. Wildest Beards The Kelce brothers couldn’t stop talking about Swift, even though she’d left
Travis hanging at the concert. Swift’s name began popping up everywhere on the podcast, including during conversations about... facial hair?
</item 46>

<item 47>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
In an interview withWSJ Magazine, published on 20 November, he described how much he admires her skills as a musician. “I’ve never been a man of words. Being
around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

However, he also acknowledged the heavy public attention around his private life as a result of his relationship, noting he’s “never dated anyone with that kind
of aura around them”. But, according to Kelce, he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life. When she acts like
that I better not be the one acting all strange.”
</item 47>

<item 48>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Since then, Swift has released four “Taylor’s Version” albums: Fearless (Taylor’s Version) in April 2021, Red (Taylor’s Version) in November 2021, Speak Now
(Taylor’s Version) in July 2023, and 1989 (Taylor’s Version) in October 2023.

“It’s not lost on me that the two great catalysts for this happening were two horrendous things that happened to me,” Swift said, when asked about the success
of her tour. “The first was getting cancelled within an inch of my life and sanity. The second was having my life’s work taken away from me by someone who hates
me.”

While Swift is arguably more famous now than ever before, the Grammy winner revealed that the constant attention and swarms of paparazzi won’t stop her from
living her life. There’s also constant rumours surrounding her personal life, specifically rumours about her relationship with Alwyn that her publicist, Tree
Paine, was forced to publicly deny.
</item 48>

<item 49>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Not to mention her rumoured relationship with Kansas City Chiefs tight end Travis Kelce caused a 400 per cent spike in NFL jersey sales.

Since her first NFL appearance in September, Swift’s occasional attendance at Kelce’s games has received pushback from NFL fans who believe her presence is
distracting from the game itself. “I’m just there to support Travis,” Swift admitted. “I have no awareness of if I’m being shown too much and pissing off a few
dads, Brads, and Chads.”

She playfully added: “Football is awesome, it turns out. I’ve been missing out my whole life.”

The “Bad Blood” singer revealed that their romance began when Kelce gave her shoutout on his podcast, admitting he attempted to give Swift his phone number
after one of her Eras Tour shows. The pair immediately began “hanging out” and by the time she attended her first Chiefs game in September, they were already a
couple. “I think some people think that they saw our first date at that game?
</item 49>

<item 50>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
This isn't fun for me'. But my second reaction was, 'Hey, that's actually a really interesting character they're writing about: She jet-sets around the world
collecting men - and she can get any of them, but she's so clingy that they leave and she cries in her marble bathtub surrounded by pearls'.

"I was like, 'I can use this'."

Lyrically, it contains some of her best lines ("darling, I'm a nightmare dressed like a daydream") and there's a delicious attention to detail - with a pen
click signifying every new entry in her little black book.

Shout-out to all the lonely Starbucks lovers; and to Chicago indie band Hard Femme who took that misheard lyric and made it a real song. Those search engine
clicks gotta be lucrative, right?
</item 50>
=== END ITEM id=677cc0f27ed06a58 ===

=== ITEM id=c0242c96c0647a6e ===
Gold sentences and facts:
Sentence 0: Kooker added later in the speech: “Music is a tremendous driver for AI technology, and AI technology presents a tremendous opportunity for the creative development of music.
  Fact 0.0: Kooker added later in the speech.
  Fact 0.1: Kooker said music is a tremendous driver for AI technology.
  Fact 0.2: Kooker said AI technology presents a tremendous opportunity for the creative development of music.
Sentence 1: Robert Kyncl, CEO of Warner Music Group – and an ex-high flyer at YouTube – commented: “With each major leap in technology, the music industry navigates a fresh set of challenges and opportunities.
  Fact 1.0: Robert Kyncl is the CEO of Warner Music Group.
  Fact 1.1: Robert Kyncl is an ex-high flyer at YouTube.
  Fact 1.2: Robert Kyncl commented that with each major leap in technology, the music industry navigates a fresh set of challenges and opportunities.

Retrieved items:
<item 1>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
He added that “platforms are quick to point to the loopholes in the law as an excuse to drag their feet or to not take the deepfakes down when requested.”

Kooker commended the No FAKES Act policy drafted by senators earlier this year, which he explained “would create a federal property right in one’s voice or
visual likeness and protect against unauthorized AI-generated replicas”.

2. Sony has ‘roughly 200 active conversations taking place’ with AI startups that “include potential equity investments”

Kooker explained that despite the negative impacts of AI, including unauthorized clones and low-quality AI music, there are still “many positive and optimistic
developments to highlight”.

He highlighted legitimate AI startups and established firms “with music ideas that want to partner with the industry”.

According to Kooker, Sony Music has “roughly 200 active conversations taking place with startups and established players” right now about building new
AI-related products and tools.
</item 1>

<item 2>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
(In our report, MBW suggested that YouTube parent Google‘s recent submission to The United States Copyright Office (USCO) – in response to a request for written
submissions as part of a study around copyright law and policy issues raised by AI systems – may have influenced Sony’s absence from Dream Track.)

Google was just one of various tech giants and AI companies that submitted responses to the USCO, amongst them, Anthropic and Stability AI.

Anthropic, in its submission to the USCO, also argued that training (large language models) LLM’s on copyrighted material is fair use.

Stability AI made a similar argument. Its position on the issue of “Fair Use” concerning AI and copyrighted content resulted in the resignation of prominent
generative AI executive Ed Newton-Rex from the company.

As reported by MBW last week, Universal Music Group submitted its response to the USCO.
</item 2>

<item 3>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
This all seemed to hint at a desire to build a ‘Content ID’-style system on YouTube to control and police content made by cloning artists’ voices;

publicly committed to three principles/pledges behind its development of music-based . These pledges promised in said tools. They promised that, amongst other
things, would help YouTube . This all seemed to hint at a desire to build a on YouTube to control and police content made by cloning artists’ voices; Then,
earlier this month, YouTube announced it was developing a specific system for music partners that allowed them to request the removal of content on its platform
that “mimics an artist’s unique singing or rapping voice”.

Evidently, all of this was enough for Universal Music Group and Warner Music Group to dip their toe into the ‘Dream Track’ experiment – though not without
caution.
</item 3>

<item 4>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
But what was maybe more surprising – for a techie like Kyncl, anyway – was his rational, level-headed explanation for why traditional music labels continue to
be relevant, and continue to thrive, in the current music ecosystem.

With the arrival of digital platforms, including open ones like YouTube and SoundCloud – and with the proliferation of indie music distribution platforms like
DistroKid and TikTok’s SoundOn – it’s inevitable that music biz observers from the world of tech like to speculating about the possibility that music labels
could, in the long-term, be headed for irrelevance.

In a world where anyone can record music using a desktop audio workstation (DAW) and upload it direct to YouTube, or to Spotify via a distribution platform,
labels have little left to do, the argument goes.

Kyncl offered strong a counterpoint to this argument at Code Media, drawing a parallel between music labels today and the advertising industry a decade or two
ago.
</item 4>

<item 5>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
That argument could yet come into play in Anthropic’s legal battle with Universal.

And given the high stakes involved – for the music industry, for the tech industry and for the shape of culture going forward – that case, along with the US
Copyright Office’s study of AI and copyright, are worth keeping a close eye on.

JKBX (pronounced "Jukebox") unlocks shared value from things people love by offering consumers access to music as an asset class — it calls them Royalty Shares.
In short: JKBX makes it possible for you to invest in music the same way you invest in stocks and other securities.Music Business Worldwide
</item 5>

<item 6>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
At the same time, we must help artists achieve their greatest creative and commercial potential…” Sir Lucian Grainge on UMG’s decision to license YouTube’s
‘Dream Track’ experiment

Robert Kyncl, CEO of Warner Music Group – and an ex-high flyer at YouTube – commented: “With each major leap in technology, the music industry navigates a fresh
set of challenges and opportunities.

“It’s not always the case that, from the outset, tech platforms partner with artists, songwriters, labels, and publishers to experiment, iterate, and find
possible solutions.

“YouTube is taking a collaborative approach with this Beta. These artists are being offered the choice to lean in, and we’re pleased to experiment and find out
what the creators come up with.”

Why, then, did Sony Music not join its rivals in granting YouTube and ‘Dream Track’ access to any of its active artists’ licensed voices?
</item 6>

<item 7>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
“No. Music is much more resilient. So first, I think what happened in the last 15 years is incredible. They had literally zero people in the subscription model
and now we have 700 million people in the world in the premium experience… which is incredible.

“And I think really credit goes to Daniel Ek for forging the path for everybody. And then companies like Apple and YouTube and Amazon following and building up
the [music] business. It’s pretty incredible what has happened.”

“You have to embrace technology, because it’s not like you can put technology in a bottle. The genie is not going back.” Robert Kyncl, Warner Music Group

Kyncl added: “I think the opportunity ahead of us is twofold. One, the continued growth in emerging markets, and also the GDPs of those countries will be rising
at the same time. So there’s lots of growth there.
</item 7>

<item 8>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
Spotify erased thousands of AI-crafted songs from its platform but also recently globally launched an AI-powered DJ that curates music for listeners while
talking to them in a synthetic voice.

“The training of generative AI using our artists’ music . . . begs the question as to which side of history all stakeholders in the music ecosystem want to be
on: the side of artists, fans and human creative expression, or on the side of deep fakes, fraud and denying artists their due compensation,” Universal Music
Group said after a song using AI to imitate Drake and The Weeknd, two of its artists, went viral.

These same conversations and contradictions are manifesting across creative industries, but artists themselves don’t always have a seat at the table.
Independent artists in particular are learning that their voices resonate louder when coming together across disciplines to push back against what Holland
describes as an “extraordinary spectrum of exploitation” that leverages their work.
</item 8>

<item 9>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
if you want a sustainable career, with repeatability and success.”

Here are three other things we learned from Kyncl’s talk at the Code Conference…

Credit: Tatiana Shepeleva / Shutterstock 1) Music will feel the impact of AI before other industries do – ‘within the next year’

Kyncl predicted that the music business will be out ahead of other industries when it comes to exploiting the potential of AI, and it will also – out of
necessity– be ahead on the issue of how to regulate and monetize the use of AI by non-rights-holders.

“Music – because it’s so broadly distributed and it’s so well aligned with the internet, because it’s short format, it lends itself to recommendations, it’s on
all platforms – is generally first in most transformations and most innovations. So it digitizes first,” he said.

“Whatever happened to the music industry 20 years ago is starting to happen in the movies and TV shows now.
</item 9>

<item 10>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Senior political figures in Washington are becoming increasingly interested in the impact of AI on the law.

On Wednesday (November 29), US Senate Majority Leader Chuck Schumer held an Artificial Intelligence (AI) Insight Forum to discuss the importance and role of
creative copyright and intellectual property in the development of AI.

Dennis Kooker, Sony Music Entertainment‘s President of Global Digital Business & US Sales, delivered a speech at the Forum, outlining the major music company’s
position on AI and copyright law.

Kooker’s public statements on the topic arrive a week after MBW pointed out that there was a notable lack of endorsement from SME, and its artists, of YouTube‘s
new experimental AI project called ‘Dream Track’ that lets creators clone the vocals of successful musicians.

The initial cohort of stars involved in the project, as we noted last week, include recording artists signed to Warner Music Group and Universal Music Group –
or labels within each of those two majors, but no Sony Music artists.
</item 10>

<item 11>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
“About the time that we started the discussion with this artist, we had begun informal talks with a generative AI start-up company whose business model focused
on working with IP rightsholders the ‘right way’.

“In other words, they respect intellectual property rights and want to work with rights holders in ways that enhance and protect the copyrighted works.”

3. SME has asked Congress to embrace a set of principles around generative AI

Kooker told the lawmakers during his speech on Wednesday that “if copyrights are protected and enforced appropriately, we are at the beginning of a multi-decade
marathon that will change the creative and commercial landscape for music”.

With that in mind, Kooker explained that Sony Music has established a set of principles to guide the company’s decision-making around generative AI.

“If copyrights are protected and enforced appropriately, we are at the beginning of a multi-decade marathon that will change the creative and commercial
landscape for music.”
</item 11>

<item 12>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
The goal is that AI tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

“It’s really interesting from a music perspective, specifically, because . . . musicians are perhaps more familiar with the idea of AI,” Holland said.
“Musicians in general are more familiar with things like music production software, and AI tools like MIDI drum loops . . . so I think that there is a certain
amount of more progressive learning from them, when it comes to technology, and its ability to make their music better.”

When it comes to art and AI, the conversation is complicated, to say the least. Musicians are nervous about industry giants copyrighting AI music and cutting
them out of the process. Major record labels are worried about AI models training on their catalogues and stealing a slice of their considerable pie.
</item 12>

<item 13>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
To be fair to Google, its submission does make note of the importance of US copyright law, including when applied to generative AI, striking “the right balance
between the legitimate interests of rightsholders and the equally legitimate interests of the public and succeeding generations of creators”.

Yet it could clearly be argued that Google’s take on copyright “harvesting” by generative AI models sits in odd contrast to the copyright-related reassurances
that YouTube has been careful to make to the music industry during the pre-development, development, and trial launch of ‘Dream Track’.

A final thought…

MBW’s sources close to Sony Music were keen to point out that the company’s general ongoing relationship with YouTube and YouTube Music is a harmonious one.

(YouTube is, after all, now the second largest commercial partner of the major record companies – with a stated ambition to catch up with Spotify in the years
ahead.
</item 13>

<item 14>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
These products range from tools for “creative or marketing assistance”, to tools that “potentially give us the ability to better protect artist content or find
it when used in an unauthorized fashion” as well as “brand new products that have never been launched before”.

Kooker said that some of these conversations “also include potential equity investments which would accelerate the development of these companies”.

The Sony Music exec cited, as an example, a recent generative AI-powered project around a reissue and remix of an album. Although Kooker didn’t name it
specifically, SME did recently announce a generative AI project with The Orb and David Gilmour, which was a partnership between Sony Music Entertainment, Legacy
Recordings AI company Vermillio

“These artists are known for their cutting-edge experimentation in music,” said Kooker.
</item 14>

<item 15>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
MBW Reacts is a series of analytical commentaries from Music Business Worldwide written in response to major recent entertainment events or news stories. MBW
Reacts is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

In recent months, Warner Music Group CEO Robert Kyncl has been keeping a close eye on AI in the music industry – not surprising, given that he came to the music
biz from ‘Big Tech’, namely YouTube, where he was Chief Business Officer.

Nor is it surprising, given his current role, that he has been pushing for price hikes at music streaming services.

Both of these topics came up during Kyncl’s Q&A session at the Code Conference in Laguna Niguel, California, on Tuesday (September 26).
</item 15>

<item 16>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
“Human authorship is a bedrock requirement of copyright.”

2. AI developers should be required to license materials and keep records of the materials they use for training

Many key players in the music industry and beyond have been adamant that using copyrighted works to train AI algorithms without authorization is a copyright
infringement, and a number of lawsuits have been launched against AI companies on these grounds.

One such lawsuit was filed by Universal Music Group, Concord Music Group and ABKCO against AI developer Anthropic, seeking potentially tens of millions of
dollars for the alleged “systematic and widespread infringement of their copyrighted song lyrics.”

But that doesn’t mean key music industry players don’t want their materials ever used to train AI – they just want AI developers to license that music, not
unlike how streaming services, bars and restaurants or movie producers license music.
</item 16>

<item 17>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
The genie is not going back.”

But he cautioned that the technology to develop a Content ID-style system for is “not yet developed, but… people are working on that.”

{Believe CEO Denis Ladegallerie might disagree. In comments earlier this year, Ladegallerie said that the tools to recognize and flag AI-generated content are
just about ready to go, and he expects to see them implemented in the coming few quarters.)

2) The streaming audio business won’t experience the same problems currently hitting streaming video

The streaming video business is going through tough times.

Netflix experienced its first-ever decline in subscriptions last year (though it has since roared back to health); the subscriber count of Disney+ appears to be
in freefall; and talk is growing about ongoing consolidation of the many streaming video services that have come online over the past several years.

Asked if the audio streaming business is likely to go through something similar, Kyncl’s answer was unequivocal.

“No.
</item 17>

<item 18>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
Especially when SME has shown its own willingness to experiment with generative AI music platforms this week via this announcement about a project with David
Gilmour, The Orb, and AI company Vermillio?

Sources close to Sony Music’s HQ in New York have suggested to MBW that the company is taking an “artist-led” approach to its experiments with any AI platform
that can manipulate the work (or voice) of its roster.

In the case of YouTube’s ‘Dream Track‘, MBW understands, SME first offered the opportunity to participate to a number of its artists, but this group of acts did
not show enthusiasm about participating.

(If one of these artists was Bad Bunny, signed to Sony’s The Orchard, you can see why: The Puerto Rican superstar expressed fury the other week about a popular
track on TikTok that had cloned his vocal stylings via AI.)
</item 18>

<item 19>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
In the meantime, music emerged out of [digitization] better and stronger and more resilient. So, I would imagine the same will happen here, which is we’ll
likely be first.”

Asked when that sea change will occur, Kyncl said: “I would say within the next year you will see lots of evolution around AI… What you will likely see is
increasing quality at a very fast pace.”

Kyncl drew an analogy between the explosion of AI use among members of the public and the explosion of user-generated content (UGC) a decade or two ago. Just as
user-generated content often infringed on copyright, so too does AI threaten the rights of artists and music rights holders.

“I would say within the next year you will see lots of evolution around AI… What you will likely see is increasing quality at a very fast pace.”
</item 19>

<item 20>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
MBW Reacts is a series of analytical commentaries from Music Business Worldwide written in response to major recent entertainment events or news stories. MBW
Reacts is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

Legislators and regulators around the world are grappling with how to address the many issues that have emerged since AI technology “went mainstream” about a
year ago.

In the US, one of the key efforts on that front is being spearheaded by the US Copyright Office, which this past summer issued a call for submissions on the
issue of AI and copyright. The goal is for the USCO to put together a study to ”help assess whether legislative or regulatory steps in this area are warranted.”

Numerous businesses involved in AI, and those with substantial copyright holdings, have submitted their thoughts on the issue, including Universal Music Group
and AI developer Anthropic.
</item 20>

<item 21>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Kooker also openly discussed Sony Music’s efforts to get unauthorized AI content removed from platforms online – including the fact that the firm has already
issued close to 10,000 separate takedowns.

Kooker further discussed Sony Music’s optimism about AI’s potential for enhancing creativity, as well as key principles the company has adopted around
generative AI.

Here are three things that stood out from Kooker’s speech…

1. Sony Music has issued close to 10,000 takedowns for unauthorized deep fakes of artists

Kooker argued in his speech that, while “the music industry sees great potential with various forms of AI technology in this early phase of AI,” so far, “the
available generative AI products are not delivering on the expected promise of new products in creative industries”.

He clarified that point by explaining that some generative AI platforms “are not expanding the business model or enhancing human creativity”.
</item 21>

<item 22>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
Yet something conspicuous is not on there: any endorsement whatsoever from the world’s second-biggest recording company, Sony Music Entertainment, or its
artists.

Why is Sony missing?

MBW understands that Sony Music (the recorded music arm of Sony Music Group) has, like Universal and Warner, been in discussions for months with YouTube over
potentially licensing the ‘Dream Track’ experiment.

During this period of discussion, YouTube has made a number of positive announcements regarding the biggest issue for any rightsholder regarding AI-driven voice
replication of artists: their ability to police it.

In August, YouTube and Universal Music Group jointly announced that the video platform was launching an ‘AI Music Incubator’ – a program by which new tools and
innovations will be developed at YouTube in close conjunction with artists and the music biz.
</item 22>

<item 23>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
So there’s lots of growth there. And then there is the price elasticity optimization in mature markets.”

That last point is a reference to the price hikes seen among all the major music streaming services over the past year or so, and the growing belief within the
music industry that streaming price hikes will be sustained – as evidenced recently by Deezer’s second price hike in a year.

Credit: QuiteSimplyStock/Shutterstock 3) The hot music rights acquisitions space is ‘likely slowing down a bit’

The second half of 2022 saw a notable slowdown in music rights acquisitions, a phenomenon that some blamed on higher interest rates and the reduced liquidity
that came with them.

Although there has been something of a rebound this year, in Kyncl’s view, there is still a slowdown in this space – though that might be good for Warner Music
Group.

“I think what happened was there was an opening where artists and songwriters were suddenly open to selling catalogs.
</item 23>

<item 24>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
And it was an incredible new revenue stream for everyone. AI is that with new super tools.”

Kyncl was referring to YouTube’s Content ID system, which scours uploaded videos for copyrighted content (video and audio), then offers the copyright owner the
option to monetize that video, or to request that it be taken down. The innovative system has essentially allowed YouTube users to upload content without paying
much attention to copyright, while ensuring that copyright owners are paid.

When it comes to the proliferation of AI, “we need to approach it with the same thoughtfulness, and we have to make sure that artists have a choice,” Kyncl
added.

And he made it clear that, in his view, simply rejecting AI and fighting against it is not an option.

“You have to embrace technology, because it’s not like you can put technology in a bottle. The genie is not going back.”
</item 24>

<item 25>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
Robert Kyncl, Warner Music Group

In Kyncl’s view, the solutions put in place to address UGC form a “blueprint” than can be used to address AI-generated content on platforms as well.

“When YouTube was formed… people started to upload content, including copyrighted material, which, obviously, put YouTube into hot water with lots of different
copyright holders. And I had the privilege of working through a lot of that and fix it up.

“But we made we made a very important decision, which was to go above and beyond the law, and build a fingerprinting software that allowed us to track the
copyright on our platform, and then have commercial relationship[s] with copyright holders to send them the money. Out of that we built a multi-billion-dollar
business, which now is a multi-billion-dollar business per year. And it was an incredible new revenue stream for everyone. AI is that with new super tools.”
</item 25>

<item 26>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Dennis Kooker, Sony Music

Kooker added later in the speech: “Music is a tremendous driver for AI technology, and AI technology presents a tremendous opportunity for the creative
development of music.

“But these opportunities must be grounded by the human creators’ vision with the machine assisting, not with the machine replacing the human creator.”

To achieve this, Kooker said that SME wants Congress to embrace the following principals:

Assure Consent, Compensation, and Credit. “New products and businesses built with music must be developed with the consent of the owner and appropriate
compensation and credit. It is essential to understand why the training of AI models is being done, what products will be developed as a result, and what the
business model is that will monetize the use of the artist’s work. Congress and the agencies should assure that creators’ rights are recognized and respected.”
Confirm That Copying Music to Train AI Models is Not Fair Use.
</item 26>

<item 27>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
Sir Lucian Grainge, Chairman and CEO of Universal Music Group, said of UMG’s decision to offer a limited license to ‘Dream Track’: “We have a fundamental
responsibility to our artists to ensure the digital ecosystem evolves to protect them and their work against unauthorized exploitation, including by generative
AI platforms.

“At the same time, we must help artists achieve their greatest creative and commercial potential – in part by providing them access to the kind of opportunities
and cutting-edge creative tools made possible by AI.”

“We have a fundamental responsibility to our artists to ensure the digital ecosystem evolves to protect them and their work against unauthorized exploitation,
including by generative AI platforms.
</item 27>

<item 28>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
They’re stars, for sure. But in both the case of Universal Music Group and Warner Music Group, they’re – currently anyway – not the biggest megastars on either
company’s books.

You might not expect, at this early stage, for the likes of UMG to put forward Drake, Taylor Swift, or The Beatles – or for WMG to put forward Ed Sheeran or Dua
Lipa – to be guinea pigs in any early AI music experiments, even if they’re run by ‘friends’ of the music industry like YouTube.

You might, then, also understandably wonder if there have been significant licensing payments made by YouTube to UMG and WMG for the right to play with the AI
vocals of artists who have signed off on ‘Dream Track’ thus far. And, if so, if these payments are being placed against any as-yet-unrecouped advances that some
of these artists may have on their label accounts.
</item 28>

<item 29>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
But last month something else happened that, we hear, effectively scuppered any chance that Sony Music would get involved with ‘Dream Track’ at this juncture.

On October 30, Google – the parent of YouTube within Alphabet Inc. – issued a filing with the US Copyright Office that outlined its core position on ‘Artificial
Intelligence and Copyright’.

The contents of this document have caused some alarm at Sony Music HQ.

What was in the Google document?

You may have read on MBW last week that Ed Newton-Rex – a generative AI pioneer, but also a published composer – decided to quit his role at Stability AI over
concerns over the company’s position on “fair use” within its own recent US Copyright Office filing.

‘Fair use’ is, as it sounds, the argument that there are cases where copyrighted material can be used and even replicated that are, well, fair enough.
</item 29>

<item 30>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
YouTube says it paid out over USD $6 billion to music rightsholders in the year to end of June 2022, with around $2 billion of that coming just from ads on
user-generated content.)

However, the October US Copyright Office filing from Google has definitely put the cat amongst the pigeons at Sony Music HQ. As we’re sure it has at the other
two major record companies.

“Any prohibition or limitation on the use of copyrighted materials for purposes of Al training would undermine the purpose of copyright and foreclose the many
opportunities that come with this technology.” Google filing with the US Copyright Office, October 30

Witness Universal Music Group’s own filing with the US Copyright Office on the topic of AI. It couldn’t offer a starker contrast to Google’s assertion that
“prohibition or limitation on the use of copyrighted materials for purposes of Al training would undermine the purpose of copyright and foreclose the many
opportunities that come with this technology”.
</item 30>
=== END ITEM id=c0242c96c0647a6e ===

=== ITEM id=bfe1345c5aec7aa5 ===
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
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 1>

<item 2>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Looking back on all that had changed around what first attracted people to Google, from the introduction of ads to the efforts to keep users within the universe
of Google products, this seemed to be the last straw.

The second major reason why Google results feel different lately was, of course, SEO — specifically, the obnoxious-kid-refusing-to-behave-in-class kind of SEO.

“SEO that goes against Google’s guidelines, it’s not new,” Ray explained. A decade ago, it used to be called “black hat” SEO, in comparison to the search
engine-approved “white hat” tactics. And Google has, as Sullivan and many SEOs told me, gotten better over time at catching SEOs playing tricks on the
algorithm.
</item 2>

<item 3>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
He realized that most search engines were just listing websites in order of how many times a search term appeared on the site and in its tags, so he focused on
stuffing his sites with keywords, resubmitting his URL to the search engines, and waiting for the results to change.

Mapstead started pulling in $25,000–$30,000 a month, working 12- to 14-hour days. “It was how long could I stay awake and how little life could I have because
this was more money than I could have ever imagined in my lifetime,” he told me. “It was like I won the lottery, and I didn’t know how long it would last.”

Around this time, in 1997, an Italian professor published a journal article about what he called Search Engines Persuasion. “Finding the right information on
the World Wide Web is becoming a fundamental problem,” he wrote. “A vast number of new companies was born just to make customer Web pages as visible as
possible,” which “has led to a bad performance degradation of search engines.”

Enter Google.
</item 3>

<item 4>
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
</item 4>

<item 5>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Quick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be the default search engine on Firefox. At
the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.

“I felt strongly that Yahoo was not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said
Yahoo had promised to reduce the number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.

“The number of users who stayed with Firefox declined noticeably during the years when Yahoo was the default,” said Baker.

Baker, who was providing a recorded deposition for Google’s defense, noted that Mozilla’s users apparently wanted and expected Google.

However, that is not the full story. Yahoo was already well behind Google in terms of search technology at that point.
</item 5>

<item 6>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
</item 7>

<item 8>
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
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Yahoo was already well behind Google in terms of search technology at that point. But also, Firefox was very well behind Chrome, which ended 2014 with nearly
50% market share and ended 2017 (when Google got prime default search position on Firefox) at nearly 65% of all web browser usage on desktop: and mobile is even
more skewed to Chrome. In other words, the number of Firefox users might have been declining for other reasons, although pushing the blame on Yahoo certainly
works in Google’s favor here!

The trial continues…
</item 9>

<item 10>
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
</item 10>

<item 11>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Enter Google. The company revolutionized search by evaluating websites based on links from other websites, seeing each link as a vote of relevance and
trustworthiness. The founders pledged to be a neutral navigation system with no ads: just a clean white screen with a search box that would bring people off of
the Google landing page and out to a helpful website as seamlessly as possible. Users quickly decided this link-based sorting methodology was superior to the
existing search engines, and by the end of 1999, Google was handling the majority of online queries.

“I was basically just spamming Facebook with cars and articles about cars and sending traffic to banner ads, and that turned into $120,000 a month.”

Mapstead, like many of the early practitioners of SEO, figured out how to adapt. Almost as soon as Google took over, a secondary market emerged for links. For a
few hundred bucks, a firm in India or the Philippines could provide thousands of links from blog networks built entirely for that purpose.
</item 11>

<item 12>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 12>

<item 13>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 13>

<item 14>
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
</item 14>

<item 15>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 15>

<item 16>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 16>

<item 17>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
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
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 19>

<item 20>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 20>

<item 21>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The process of this test did remind me how much data Google collects — not just my searches but anything I say to a smart speaker, any search result or ad that
I click — which will be a point in favour of DuckDuckGo for some, even if it didn’t perform as well and is harder to read. The data collection does also mean
that Google (not just search but also mobile widgets and other services) personalises to me more effectively. Loading Bing has overhauled its layout over the
past year to be very graphic and augmented by generative AI explanations from its chatbot. I like the idea of being able to ask follow-up questions in searches
this way, but in practice it’s not really there yet. In a search for “kids BMX bike green” (Google did pretty well and DuckDuckGo ignored the green part), Bing
Chat offered a comparison of a few different options but they were too expensive. I asked, “any under $200?” and it replied with more options that were all over
$300.
</item 21>

<item 22>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 22>

<item 23>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 23>

<item 24>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Google’s software is constantly crawling the web and updating its central index, which itself is more than 100 million gigabytes in size. When you enter a
query, Search uses a few quick processes to decide the meaning of your question, including what kind of information you might be after. Then, beyond just
looking for sites with the words you typed in, the system uses machine learning to rank results according to what it thinks you’re looking for, prioritising
high quality and easy to use sites. If you allow it, Google does bring in some information it knows about you to further refine results, including your location
and web history. If you’re looking to turn this off, it’s in your Google account settings called “Web & App Activity”. Google says it does not infer personal
information like race or religion to shape results. Loading Google works with independent “search quality raters” to gauge the effectiveness of its processes,
and says it ran almost 900,000 quality tests in 2022.
</item 24>

<item 25>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 25>

<item 26>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
“Maybe you ask a friend.”

After hearing so much about what it was like to be an SEO, I decided it was time to better understand what’s been going on from the perspective of the search
engine. Google was slow to allow someone to talk with me, possibly because of the giant PR clusterfuck that has been the company’s past year (accused by the
federal government of being a monopoly; increasingly despised by the public; losing ground to Reddit, TikTok, and large language models), so I decided to start
by meeting up with a chipper, charismatic man named Duane Forrester.

Forrester was at Microsoft from 2007 until 2015, where he helped launch and manage Bing, the perpetual underdog to Google’s domination of online search. Before
and after his time at Microsoft, Forrester worked as an SEO, so he sees the industry from both sides, like an aerospace engineer who spent a few years at the
Department of Defense, left for the private sector, and now is much better at winning military contracts.
</item 26>

<item 27>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 27>

<item 28>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 28>

<item 29>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
the list of blue links in search results that are not sponsored. “Ad keyword matching is a long-standing and well-known process that is designed to connect
people to relevant ads. A separate process, which has nothing to do with ads, is used to match organic results to a query,” he said. “It’s no secret that Google
Search looks beyond the specific words in a query to better understand their meaning ... If you make a spelling mistake, or search for a term that’s not on a
page but where the page has a close synonym, or if you aren’t even sure exactly how to search for something, our meaning matching systems help.” It is true that
some search queries result in more ads than others, and what ads you get can vary wildly depending on your request, which is the result of the keyword matching
Google uses to sell ads. But Google denies pushing users towards the most monetisable subjects.
</item 29>

<item 30>
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
</item 30>
=== END ITEM id=bfe1345c5aec7aa5 ===
